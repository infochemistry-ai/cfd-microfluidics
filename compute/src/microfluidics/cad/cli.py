"""Command line entry point for configuration-driven CAD mesh generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from microfluidics.cad.config import load_geometry_pipeline_config
from microfluidics.mesh.mesh_builder import build_mesh_from_geometry_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gmsh", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_geometry_pipeline_config(args.config)
    result = build_mesh_from_geometry_config(
        config,
        args.output_dir,
        gmsh_executable=args.gmsh,
    )
    if result.cad_mesh is not None:
        summary = {
            "source_kind": result.source_kind,
            "step": str(result.cad_mesh.artifacts.step_path),
            "brep": str(result.cad_mesh.artifacts.brep_path),
            "geo": str(result.cad_mesh.artifacts.geo_path),
            "msh": str(result.cad_mesh.generation.msh_path),
            "tetra_count": int(result.cad_mesh.mesh.tetrahedra.shape[0]),
            "valid": result.cad_mesh.validation.is_valid,
        }
    else:
        assert result.legacy_mesh is not None
        summary = {
            "source_kind": result.source_kind,
            "legacy_cell_count": int(result.legacy_mesh.cells.shape[0]),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
