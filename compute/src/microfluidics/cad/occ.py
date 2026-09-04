"""Small, explicit pythonocc-core boundary for microfluidic CAD operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from microfluidics.cad.config import CadParameters


class OpenCascadeUnavailableError(RuntimeError):
    """Raised when the optional conda-provided pythonocc-core runtime is absent."""


class CadBuildError(RuntimeError):
    """Raised when OpenCASCADE cannot construct or exchange a valid solid."""


@dataclass(frozen=True)
class CadShape:
    shape: Any
    bbox: tuple[float, float, float, float, float, float]
    volume: float
    solid_count: int


def require_pythonocc() -> None:
    try:
        import OCC.Core  # noqa: F401
    except ImportError as exc:
        raise OpenCascadeUnavailableError(
            "pythonocc-core 7.9.3 is required for CAD geometry. Install the "
            "conda environment from compute/environment.cad.yml; the PyPI-only "
            "uv environment intentionally cannot provide pythonocc-core."
        ) from exc


def _bbox(shape: Any) -> tuple[float, float, float, float, float, float]:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.AddOptimal(shape, box, False, False)
    return tuple(float(value) for value in box.Get())  # type: ignore[return-value]


def _solid_count(shape: Any) -> int:
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer

    count = 0
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def inspect_shape(shape: Any) -> CadShape:
    require_pythonocc()
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    if shape is None or shape.IsNull():
        raise CadBuildError("OpenCASCADE returned a null shape")
    if not BRepCheck_Analyzer(shape).IsValid():
        raise CadBuildError("OpenCASCADE shape is not topologically valid")
    solid_count = _solid_count(shape)
    if solid_count != 1:
        raise CadBuildError(
            f"CAD fluid volume must contain exactly one solid, got {solid_count}"
        )
    properties = GProp_GProps()
    brepgprop.VolumeProperties(shape, properties)
    volume = float(properties.Mass())
    if volume <= 0:
        raise CadBuildError(f"CAD fluid solid has non-positive volume: {volume}")
    return CadShape(
        shape=shape, bbox=_bbox(shape), volume=volume, solid_count=solid_count
    )


def ensure_tessellatable(cad_shape: CadShape) -> None:
    """Verify that OCCT can triangulate the shape for viewer rendering."""

    require_pythonocc()
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

    xmin, ymin, zmin, xmax, ymax, zmax = cad_shape.bbox
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    mesher = BRepMesh_IncrementalMesh(
        cad_shape.shape,
        max(span * 1e-3, 1e-8),
        False,
        0.5,
        True,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise CadBuildError("OpenCASCADE could not tessellate STEP for viewer display")


def _make_box(
    xmin: float,
    ymin: float,
    zmin: float,
    dx: float,
    dy: float,
    dz: float,
) -> Any:
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt

    maker = BRepPrimAPI_MakeBox(gp_Pnt(xmin, ymin, zmin), dx, dy, dz)
    maker.Build()
    if not maker.IsDone():
        raise CadBuildError("OpenCASCADE failed to create a channel primitive")
    return maker.Shape()


def _fuse_pair(left: Any, right: Any) -> Any:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

    operation = BRepAlgoAPI_Fuse(left, right)
    operation.Build()
    if not operation.IsDone():
        raise CadBuildError("BRepAlgoAPI_Fuse failed while joining fluid components")
    operation.SimplifyResult(True, True)
    return operation.Shape()


def _fuse_all(shapes: Iterable[Any]) -> Any:
    iterator = iter(shapes)
    try:
        result = next(iterator)
    except StopIteration as exc:
        raise CadBuildError("no fluid primitives were provided") from exc
    for shape in iterator:
        result = _fuse_pair(result, shape)
    return result


def _edge_is_on_terminal(edge: Any, params: CadParameters) -> bool:
    bounds = _bbox(edge)
    tolerance = max(params.expected_bbox()[3] - params.expected_bbox()[0], 1.0) * 1e-10
    xmin, ymin, _zmin, xmax, ymax, _zmax = bounds
    return (
        (
            abs(xmin + params.inlet_length) <= tolerance
            and abs(xmax + params.inlet_length) <= tolerance
        )
        or (
            abs(xmin - params.inlet_length) <= tolerance
            and abs(xmax - params.inlet_length) <= tolerance
        )
        or (
            abs(ymin - params.outlet_length) <= tolerance
            and abs(ymax - params.outlet_length) <= tolerance
        )
    )


def _fillet_walls(shape: Any, params: CadParameters) -> Any:
    from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    builder = BRepFilletAPI_MakeFillet(shape)
    edge_count = 0
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        if not _edge_is_on_terminal(edge, params):
            builder.Add(params.fillet_radius, edge)
            edge_count += 1
        explorer.Next()
    if edge_count == 0:
        raise CadBuildError("no wall edges were eligible for filleting")
    builder.Build()
    if not builder.IsDone():
        raise CadBuildError(
            f"OpenCASCADE could not apply fillet_radius={params.fillet_radius}"
        )
    return builder.Shape()


def build_tjunction(params: CadParameters) -> CadShape:
    """Build one fused BRep solid representing the fluid volume."""

    params.validate()
    require_pythonocc()
    half_inlet = 0.5 * params.inlet_width
    half_outlet = 0.5 * params.outlet_width
    primitives = [
        _make_box(
            -params.inlet_length,
            -half_inlet,
            0.0,
            params.inlet_length,
            params.inlet_width,
            params.channel_height,
        ),
        _make_box(
            0.0,
            -half_inlet,
            0.0,
            params.inlet_length,
            params.inlet_width,
            params.channel_height,
        ),
        _make_box(
            -half_outlet,
            0.0,
            0.0,
            params.outlet_width,
            params.outlet_length,
            params.channel_height,
        ),
    ]
    if params.include_cavities:
        primitives.extend(
            (
                _make_box(
                    -half_outlet - params.cavity_depth,
                    params.cavity_offset_from_junction,
                    0.0,
                    params.cavity_depth,
                    params.cavity_length,
                    params.channel_height,
                ),
                _make_box(
                    half_outlet,
                    params.cavity_offset_from_junction,
                    0.0,
                    params.cavity_depth,
                    params.cavity_length,
                    params.channel_height,
                ),
            )
        )
    shape = _fuse_all(primitives)
    if params.fillet_radius > 0:
        shape = _fillet_walls(shape, params)
    return inspect_shape(shape)


def write_brep(cad_shape: CadShape, path: str | Path) -> Path:
    require_pythonocc()
    from OCC.Core.BRepTools import breptools

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not breptools.Write(cad_shape.shape, str(output)) or not output.is_file():
        raise CadBuildError(f"failed to write BREP: {output}")
    return output


def write_step(cad_shape: CadShape, path: str | Path) -> Path:
    require_pythonocc()
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import (
        STEPControl_AsIs,
        STEPControl_Controller,
        STEPControl_Writer,
    )

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not STEPControl_Controller.Init():
        raise CadBuildError("failed to initialize OpenCASCADE STEP controller")
    if not Interface_Static.SetCVal("xstep.cascade.unit", "M"):
        raise CadBuildError("failed to set OpenCASCADE internal STEP unit to metres")
    if not Interface_Static.SetCVal("write.step.unit", "M"):
        raise CadBuildError("failed to set STEP output unit to metres")
    writer = STEPControl_Writer()
    transfer_status = writer.Transfer(cad_shape.shape, STEPControl_AsIs)
    if transfer_status != IFSelect_RetDone:
        raise CadBuildError(f"failed to transfer shape to STEP writer: {output}")
    if writer.Write(str(output)) != IFSelect_RetDone or not output.is_file():
        raise CadBuildError(f"failed to write STEP: {output}")
    return output


def read_step(path: str | Path) -> CadShape:
    require_pythonocc()
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import STEPControl_Controller, STEPControl_Reader

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not STEPControl_Controller.Init():
        raise CadBuildError("failed to initialize OpenCASCADE STEP controller")
    if not Interface_Static.SetCVal("xstep.cascade.unit", "M"):
        raise CadBuildError("failed to set OpenCASCADE internal STEP unit to metres")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise CadBuildError(f"OpenCASCADE could not read STEP: {source}")
    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise CadBuildError(f"STEP contains no transferable roots: {source}")
    return inspect_shape(reader.OneShape())


__all__ = [
    "CadBuildError",
    "CadShape",
    "OpenCascadeUnavailableError",
    "build_tjunction",
    "ensure_tessellatable",
    "inspect_shape",
    "read_step",
    "require_pythonocc",
    "write_brep",
    "write_step",
]
