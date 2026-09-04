"""VTK export utilities for mesh-oriented T-junction scaffolds."""

from pathlib import Path
from typing import Dict

import numpy as np

from microfluidics.mesh.mesh_builder import BOUNDARY_TAG_IDS, MeshDataModel


def _vtk_dtype_name(array: np.ndarray) -> str:
    if np.issubdtype(array.dtype, np.floating):
        if array.dtype == np.float32:
            return "Float32"
        return "Float64"
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if array.dtype == np.uint8:
            return "UInt8"
        if array.dtype == np.uint16:
            return "UInt16"
        return "UInt32"
    if np.issubdtype(array.dtype, np.integer):
        if array.dtype == np.int32:
            return "Int32"
        return "Int64"
    raise TypeError(f"Unsupported VTK dtype: {array.dtype}")


def _as_vtk_data_array(
    name: str,
    array: np.ndarray,
    n_components: int | None = None,
) -> str:
    arr = np.asarray(array)
    if arr.ndim == 1:
        flat = arr
        components = n_components or 1
    elif arr.ndim == 2:
        components = n_components or arr.shape[1]
        flat = arr.reshape(-1)
    else:
        raise ValueError(f"Unsupported array rank for VTK export: {arr.ndim}")
    data_type = _vtk_dtype_name(arr)
    values = " ".join(str(value) for value in flat.tolist())
    component_attr = f' NumberOfComponents="{components}"' if components > 1 else ""
    return (
        f'<DataArray type="{data_type}" Name="{name}"{component_attr} format="ascii">'
        f"{values}</DataArray>"
    )


def export_volume_vtu(
    mesh: MeshDataModel,
    output_path: Path,
    point_fields: Dict[str, np.ndarray] | None = None,
    cell_fields: Dict[str, np.ndarray] | None = None,
) -> Path:
    """Export volume mesh with connectivity and cell/point fields as VTU."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    if cells.ndim != 2:
        raise ValueError("mesh.cells must be a 2D array with explicit connectivity.")

    connectivity = cells.reshape(-1)
    offsets = np.arange(1, cells.shape[0] + 1, dtype=np.int64) * cells.shape[1]
    cell_types = np.asarray(mesh.cell_types, dtype=np.uint8)
    point_data = dict(mesh.point_data)
    if point_fields:
        point_data.update(point_fields)
    export_cell_data = dict(mesh.cell_data)
    if cell_fields:
        export_cell_data.update(cell_fields)

    point_section = "\n".join(
        _as_vtk_data_array(name, np.asarray(values))
        for name, values in point_data.items()
    )
    cell_section = "\n".join(
        _as_vtk_data_array(name, np.asarray(values))
        for name, values in export_cell_data.items()
    )
    content = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{cells.shape[0]}">
      <PointData>
{point_section}
      </PointData>
      <CellData>
{cell_section}
      </CellData>
      <Points>
        {_as_vtk_data_array("Points", points, n_components=3)}
      </Points>
      <Cells>
        {_as_vtk_data_array("connectivity", connectivity)}
        {_as_vtk_data_array("offsets", offsets)}
        {_as_vtk_data_array("types", cell_types)}
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _triangulate_quads(quads: np.ndarray) -> np.ndarray:
    if quads.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    tri = np.zeros((quads.shape[0] * 2, 3), dtype=np.int64)
    tri[0::2] = quads[:, [0, 1, 2]]
    tri[1::2] = quads[:, [0, 2, 3]]
    return tri


def export_boundary_surfaces_vtp(
    mesh: MeshDataModel,
    output_dir: Path,
    file_prefix: str,
) -> Dict[str, Path]:
    """Export boundary surfaces to per-tag VTP files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}
    for tag_name, quads in mesh.boundary_faces.items():
        quads_arr = np.asarray(quads, dtype=np.int64)
        triangles = _triangulate_quads(quads_arr)
        if triangles.size == 0:
            continue
        used_point_ids = np.unique(triangles.reshape(-1))
        point_map = -np.ones(mesh.points.shape[0], dtype=np.int64)
        point_map[used_point_ids] = np.arange(used_point_ids.size, dtype=np.int64)
        remapped_triangles = point_map[triangles]
        points = mesh.points[used_point_ids]
        connectivity = remapped_triangles.reshape(-1)
        offsets = np.arange(1, remapped_triangles.shape[0] + 1, dtype=np.int64) * 3
        boundary_id = BOUNDARY_TAG_IDS.get(tag_name, 0)
        boundary_field = np.full(
            remapped_triangles.shape[0], boundary_id, dtype=np.int32
        )
        surface_path = output_dir / f"{file_prefix}_{tag_name}.vtp"
        content = f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{points.shape[0]}" NumberOfPolys="{remapped_triangles.shape[0]}">
      <PointData></PointData>
      <CellData>
        {_as_vtk_data_array("boundary_tag_id", boundary_field)}
      </CellData>
      <Points>
        {_as_vtk_data_array("Points", points, n_components=3)}
      </Points>
      <Polys>
        {_as_vtk_data_array("connectivity", connectivity)}
        {_as_vtk_data_array("offsets", offsets)}
      </Polys>
    </Piece>
  </PolyData>
</VTKFile>
"""
        surface_path.write_text(content, encoding="utf-8")
        exported[tag_name] = surface_path
    return exported


def export_mesh_bundle(
    mesh: MeshDataModel,
    output_dir: Path,
    file_prefix: str,
    export_volume: bool = True,
    export_surfaces: bool = True,
    point_fields: Dict[str, np.ndarray] | None = None,
    cell_fields: Dict[str, np.ndarray] | None = None,
) -> Dict[str, object]:
    """Export mesh scaffold bundle in ParaView-friendly VTK formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, object] = {}
    if export_volume:
        result["volume_vtu"] = str(
            export_volume_vtu(
                mesh=mesh,
                output_path=output_dir / f"{file_prefix}.vtu",
                point_fields=point_fields,
                cell_fields=cell_fields,
            )
        )
    if export_surfaces:
        boundary_files = export_boundary_surfaces_vtp(
            mesh=mesh,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )
        result["boundary_vtp"] = {
            name: str(path) for name, path in boundary_files.items()
        }
    return result
