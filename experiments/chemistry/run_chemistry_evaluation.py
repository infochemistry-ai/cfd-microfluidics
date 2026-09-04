"""Evaluate a chemistry mechanism at one homogeneous state.

This entrypoint is intentionally independent from the Gmsh manifest pipeline.
It does not read a mesh, velocity, transport field or temperature field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for path in (PROJECT_ROOT, COMPUTE_SRC):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from microfluidics.chemistry import (  # noqa: E402
    ChemistryError,
    compile_mechanism,
    load_mechanism,
)


def _concentration(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("concentration must use SPECIES=VALUE")
    name, raw_number = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("concentration species cannot be empty")
    try:
        number = float(raw_number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"concentration for {name!r} must be numeric"
        ) from exc
    return name, number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--temperature-k", type=float, required=True)
    parser.add_argument("--pressure-pa", type=float, default=101325.0)
    parser.add_argument(
        "--concentration",
        type=_concentration,
        action="append",
        default=[],
        metavar="SPECIES=VALUE",
        help="Species concentration in mol/m^3; may be repeated.",
    )
    parser.add_argument(
        "--allow-missing-enthalpy",
        action="store_true",
        help="Evaluate kinetic/species sources and emit null heat release when dH is incomplete.",
    )
    parser.add_argument(
        "--allow-unbalanced-molecular-weights",
        action="store_true",
        help="Disable compile-time molecular-weight balance rejection for diagnostic mechanisms.",
    )
    parser.add_argument(
        "--allow-outside-authored-range",
        action="store_true",
        help="Disable authored temperature/pressure range enforcement.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. The result is always printed to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    concentrations: dict[str, float] = {}
    for name, value in args.concentration:
        if name in concentrations:
            parser.error(f"duplicate concentration for species {name!r}")
        concentrations[name] = value
    if not concentrations:
        parser.error("at least one --concentration is required")

    try:
        mechanism = load_mechanism(args.mechanism)
        compiled = compile_mechanism(
            mechanism,
            require_reaction_enthalpy=not args.allow_missing_enthalpy,
            validate_mass_balance=not args.allow_unbalanced_molecular_weights,
            strict_temperature_range=not args.allow_outside_authored_range,
            strict_pressure_range=not args.allow_outside_authored_range,
        )
        result = compiled.evaluate(
            concentrations,
            temperature_k=args.temperature_k,
            pressure_pa=args.pressure_pa,
            require_heat_release=not args.allow_missing_enthalpy,
        ).to_dict()
    except ChemistryError as exc:
        parser.error(str(exc))

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
