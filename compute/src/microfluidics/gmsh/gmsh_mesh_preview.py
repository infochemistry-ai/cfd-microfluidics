"""Preview and VTK export helpers for imported external Gmsh tetra meshes."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh

VTK_TETRA = 10


def _vtk_dtype_name(array: np.ndarray) -> str:
    if np.issubdtype(array.dtype, np.floating):
        return "Float64" if array.dtype != np.float32 else "Float32"
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if array.dtype == np.uint8:
            return "UInt8"
        if array.dtype == np.uint16:
            return "UInt16"
        return "UInt32"
    if np.issubdtype(array.dtype, np.integer):
        return "Int64" if array.dtype != np.int32 else "Int32"
    raise TypeError(f"Unsupported VTK dtype: {array.dtype}")


def _vtk_data_array(
    name: str, array: np.ndarray, n_components: int | None = None
) -> str:
    arr = np.asarray(array)
    if arr.ndim == 1:
        components = 1 if n_components is None else n_components
        flat = arr
    elif arr.ndim == 2:
        components = arr.shape[1] if n_components is None else n_components
        flat = arr.reshape(-1)
    else:
        raise ValueError("VTK data array supports only rank-1 or rank-2 arrays.")
    values = " ".join(str(value) for value in flat.tolist())
    component_attr = f' NumberOfComponents="{components}"' if components > 1 else ""
    dtype = _vtk_dtype_name(arr)
    return (
        f'<DataArray type="{dtype}" Name="{name}"{component_attr} format="ascii">'
        f"{values}</DataArray>"
    )


def export_imported_mesh_vtu(mesh: ImportedTetraMesh, output_path: Path) -> Path:
    """Export tetra volume mesh with basic cell fields to VTU."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = mesh.points.astype(np.float64, copy=False)
    tets = mesh.tetrahedra.astype(np.int64, copy=False)
    connectivity = tets.reshape(-1)
    offsets = np.arange(1, tets.shape[0] + 1, dtype=np.int64) * 4
    cell_types = np.full((tets.shape[0],), VTK_TETRA, dtype=np.uint8)
    content = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{tets.shape[0]}">
      <PointData></PointData>
      <CellData>
        {_vtk_data_array("cell_volume", mesh.cell_volumes.astype(np.float64))}
      </CellData>
      <Points>
        {_vtk_data_array("Points", points, n_components=3)}
      </Points>
      <Cells>
        {_vtk_data_array("connectivity", connectivity)}
        {_vtk_data_array("offsets", offsets)}
        {_vtk_data_array("types", cell_types)}
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _boundary_group_ids(mesh: ImportedTetraMesh) -> np.ndarray:
    boundary_count = mesh.boundary_face_indices.shape[0]
    group_ids = np.zeros((boundary_count,), dtype=np.int32)
    boundary_to_row = {
        int(face_id): idx
        for idx, face_id in enumerate(mesh.boundary_face_indices.tolist())
    }
    for face_id in mesh.wall_faces.tolist():
        row = boundary_to_row.get(int(face_id))
        if row is not None:
            group_ids[row] = 1
    for face_id in mesh.inlet_faces.tolist():
        row = boundary_to_row.get(int(face_id))
        if row is not None:
            group_ids[row] = 2
    for face_id in mesh.outlet_faces.tolist():
        row = boundary_to_row.get(int(face_id))
        if row is not None:
            group_ids[row] = 3
    for face_id in mesh.boundary_unresolved_faces.tolist():
        row = boundary_to_row.get(int(face_id))
        if row is not None:
            group_ids[row] = 4
    return group_ids


def _export_boundary_subset_vtp(
    *,
    mesh: ImportedTetraMesh,
    face_indices: np.ndarray,
    output_path: Path,
    include_group_ids: bool,
) -> Path | None:
    if face_indices.size == 0:
        return None
    tris = mesh.face_vertices[face_indices]
    used_points = np.unique(tris.reshape(-1))
    point_map = -np.ones((mesh.points.shape[0],), dtype=np.int64)
    point_map[used_points] = np.arange(used_points.shape[0], dtype=np.int64)
    remapped = point_map[tris]
    connectivity = remapped.reshape(-1)
    offsets = np.arange(1, remapped.shape[0] + 1, dtype=np.int64) * 3

    points = mesh.points[used_points]
    tag_values = mesh.boundary_tag_per_face[face_indices].astype(np.int32, copy=False)
    cell_data_rows = [_vtk_data_array("boundary_tag", tag_values)]
    if include_group_ids:
        group_ids = _boundary_group_ids(mesh)
        row_map = {
            int(fid): i for i, fid in enumerate(mesh.boundary_face_indices.tolist())
        }
        subset_group = np.asarray(
            [group_ids[row_map[int(fid)]] for fid in face_indices],
            dtype=np.int32,
        )
        cell_data_rows.append(_vtk_data_array("boundary_group_id", subset_group))
    cell_data = "\n        ".join(cell_data_rows)

    content = f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{points.shape[0]}" NumberOfPolys="{remapped.shape[0]}">
      <PointData></PointData>
      <CellData>
        {cell_data}
      </CellData>
      <Points>
        {_vtk_data_array("Points", points, n_components=3)}
      </Points>
      <Polys>
        {_vtk_data_array("connectivity", connectivity)}
        {_vtk_data_array("offsets", offsets)}
      </Polys>
    </Piece>
  </PolyData>
</VTKFile>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def export_imported_mesh_boundary_vtp(
    mesh: ImportedTetraMesh,
    output_dir: Path,
    file_prefix: str,
) -> Dict[str, str]:
    """Export combined and per-group boundary VTP previews."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exports: Dict[str, str] = {}

    all_boundary_path = output_dir / f"{file_prefix}_boundary_all.vtp"
    all_written = _export_boundary_subset_vtp(
        mesh=mesh,
        face_indices=mesh.boundary_face_indices,
        output_path=all_boundary_path,
        include_group_ids=True,
    )
    if all_written is not None:
        exports["boundary_all"] = str(all_written)

    for name, face_indices in (
        ("boundary_wall", mesh.wall_faces),
        ("boundary_inlet", mesh.inlet_faces),
        ("boundary_outlet", mesh.outlet_faces),
        ("boundary_unresolved", mesh.boundary_unresolved_faces),
    ):
        out_path = output_dir / f"{file_prefix}_{name}.vtp"
        written = _export_boundary_subset_vtp(
            mesh=mesh,
            face_indices=face_indices,
            output_path=out_path,
            include_group_ids=False,
        )
        if written is not None:
            exports[name] = str(written)
    return exports


def _sample_indices(n_items: int, max_points: int) -> np.ndarray:
    if n_items <= max_points:
        return np.arange(n_items, dtype=np.int64)
    stride = max(1, n_items // max_points)
    return np.arange(0, n_items, stride, dtype=np.int64)[:max_points]


def save_import_previews(
    mesh: ImportedTetraMesh,
    output_dir: Path,
    file_prefix: str,
    *,
    max_points: int = 60000,
) -> Dict[str, str]:
    """Save quick matplotlib previews for debug/inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    previews: Dict[str, str] = {}

    cell_idx = _sample_indices(mesh.cell_centers.shape[0], max_points)
    cell_xy = mesh.cell_centers[cell_idx, :2]
    fig_xy, ax_xy = plt.subplots(figsize=(7.0, 6.0))
    ax_xy.scatter(
        cell_xy[:, 0], cell_xy[:, 1], s=0.4, c="tab:blue", alpha=0.6, linewidths=0
    )
    ax_xy.set_title("Tetra Cell Centers (XY)")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    fig_xy.tight_layout()
    cell_xy_path = output_dir / f"{file_prefix}_cell_centers_xy.png"
    fig_xy.savefig(cell_xy_path, dpi=180)
    plt.close(fig_xy)
    previews["cell_centers_xy"] = str(cell_xy_path)

    boundary_face_idx = mesh.boundary_face_indices
    if boundary_face_idx.size > 0:
        b_sample_rows = _sample_indices(boundary_face_idx.shape[0], max_points)
        sampled_faces = boundary_face_idx[b_sample_rows]
        centers = mesh.face_centers[sampled_faces]
        colors = np.zeros((sampled_faces.shape[0],), dtype=np.int32)
        wall_set = set(mesh.wall_faces.tolist())
        inlet_set = set(mesh.inlet_faces.tolist())
        outlet_set = set(mesh.outlet_faces.tolist())
        unresolved_set = set(mesh.boundary_unresolved_faces.tolist())
        for row, face_id in enumerate(sampled_faces.tolist()):
            if face_id in wall_set:
                colors[row] = 1
            elif face_id in inlet_set:
                colors[row] = 2
            elif face_id in outlet_set:
                colors[row] = 3
            elif face_id in unresolved_set:
                colors[row] = 4
            else:
                colors[row] = 0

        cmap = plt.get_cmap("tab10")
        fig_b, ax_b = plt.subplots(figsize=(7.0, 6.0))
        ax_b.scatter(
            centers[:, 0],
            centers[:, 1],
            s=1.5,
            c=colors,
            cmap=cmap,
            alpha=0.85,
            linewidths=0,
        )
        ax_b.set_title("Boundary Face Groups (XY)")
        ax_b.set_xlabel("x [m]")
        ax_b.set_ylabel("y [m]")
        ax_b.set_aspect("equal", adjustable="box")
        fig_b.tight_layout()
        b_path = output_dir / f"{file_prefix}_boundary_groups_xy.png"
        fig_b.savefig(b_path, dpi=180)
        plt.close(fig_b)
        previews["boundary_groups_xy"] = str(b_path)

    return previews
