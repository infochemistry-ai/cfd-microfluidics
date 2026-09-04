from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.gmsh.tetra.gmsh_tetra_regime_guardrail import (
    build_scalar_regime_audit,
)


def _one_cell_mesh():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    boundary_triangles = np.asarray(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    boundary_face_tags = np.asarray([1, 3, 4, 4], dtype=np.int32)
    return build_imported_tetra_mesh(
        source_path=Path("regime_guardrail_tiny.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data={
            "left_inlet": np.asarray([1, 2], dtype=np.int32),
            "outlet": np.asarray([3, 2], dtype=np.int32),
            "walls": np.asarray([4, 2], dtype=np.int32),
        },
    )


def _face_velocity_for_outlet_flux(mesh, flux: float) -> np.ndarray:
    out = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    outlet_face = int(mesh.outlet_faces[0])
    out[outlet_face] = float(flux) / float(mesh.face_areas[outlet_face])
    return out


def test_mass_regime_supported_at_grid_peclet_and_schmidt_threshold() -> None:
    mesh = _one_cell_mesh()
    h = float(mesh.cell_volumes[0]) ** (1.0 / 3.0)
    diffusivity = 1e-6
    face_normal_velocity = _face_velocity_for_outlet_flux(mesh, 2.0 * diffusivity * h)

    audit = build_scalar_regime_audit(
        mesh,
        face_normal_velocity,
        diffusivity=diffusivity,
        kinematic_viscosity=1000.0 * diffusivity,
        scalar_kind="mass",
        max_grid_peclet=2.0,
        max_schmidt=1000.0,
    )

    assert audit["supported_accuracy"] is True
    assert audit["support_status"] == "supported"
    assert float(audit["grid_peclet"]["max"]) == pytest.approx(2.0)
    assert float(audit["inputs"]["schmidt_number"]) == pytest.approx(1000.0)


def test_mass_regime_reports_high_grid_peclet_as_warning() -> None:
    mesh = _one_cell_mesh()
    h = float(mesh.cell_volumes[0]) ** (1.0 / 3.0)
    diffusivity = 1e-6
    face_normal_velocity = _face_velocity_for_outlet_flux(mesh, 2.1 * diffusivity * h)

    audit = build_scalar_regime_audit(
        mesh,
        face_normal_velocity,
        diffusivity=diffusivity,
        kinematic_viscosity=diffusivity,
        scalar_kind="mass",
        max_grid_peclet=2.0,
        max_schmidt=1000.0,
    )

    assert audit["supported_accuracy"] is False
    assert audit["blocking_error"] is False
    assert audit["severity"] == "warning"
    assert audit["warning_codes"] == ["warning_high_grid_peclet"]
    assert audit["error_codes"] == []


def test_mass_regime_reports_high_schmidt_as_warning() -> None:
    mesh = _one_cell_mesh()
    face_normal_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)

    audit = build_scalar_regime_audit(
        mesh,
        face_normal_velocity,
        diffusivity=3e-10,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
        max_grid_peclet=2.0,
        max_schmidt=1000.0,
    )

    assert audit["supported_accuracy"] is False
    assert audit["blocking_error"] is False
    assert audit["warning_codes"] == ["warning_high_schmidt"]


def test_thermal_regime_uses_prandtl_not_schmidt() -> None:
    mesh = _one_cell_mesh()
    face_normal_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)

    audit = build_scalar_regime_audit(
        mesh,
        face_normal_velocity,
        diffusivity=1e-8,
        kinematic_viscosity=2e-6,
        scalar_kind="thermal",
        max_grid_peclet=2.0,
        max_prandtl=100.0,
    )

    assert audit["supported_accuracy"] is False
    assert audit["blocking_error"] is False
    assert audit["warning_codes"] == ["warning_high_prandtl"]
    assert "prandtl_number" in audit["inputs"]
    assert "schmidt_number" not in audit["inputs"]


def test_dimensionless_ratio_canonical_contract_is_scalar_kind_independent() -> None:
    mesh = _one_cell_mesh()
    face_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    mass = build_scalar_regime_audit(
        mesh,
        face_velocity,
        diffusivity=1e-6,
        kinematic_viscosity=2e-6,
        scalar_kind="mass",
    )
    thermal = build_scalar_regime_audit(
        mesh,
        face_velocity,
        diffusivity=1e-6,
        kinematic_viscosity=2e-6,
        scalar_kind="thermal",
    )

    for audit, name, symbol in (
        (mass, "schmidt_number", "Sc"),
        (thermal, "prandtl_number", "Pr"),
    ):
        assert audit["inputs"]["dimensionless_ratio"] == pytest.approx(2.0)
        assert audit["inputs"]["dimensionless_ratio_name"] == name
        assert audit["inputs"]["dimensionless_ratio_symbol"] == symbol
        assert audit["checks"]["dimensionless_ratio_within_warning_threshold"] is True
        assert audit["checks"][f"{symbol.lower()}_supported"] is True


def test_mass_regime_numpy_and_torch_input_parity() -> None:
    torch = pytest.importorskip("torch")

    mesh = _one_cell_mesh()
    h = float(mesh.cell_volumes[0]) ** (1.0 / 3.0)
    diffusivity = 1e-6
    face_normal_velocity = _face_velocity_for_outlet_flux(mesh, 1.5 * diffusivity * h)

    numpy_audit = build_scalar_regime_audit(
        mesh,
        face_normal_velocity,
        diffusivity=diffusivity,
        kinematic_viscosity=diffusivity,
        scalar_kind="mass",
        max_grid_peclet=2.0,
        max_schmidt=1000.0,
    )
    torch_audit = build_scalar_regime_audit(
        mesh,
        torch.as_tensor(face_normal_velocity, dtype=torch.float64),
        diffusivity=diffusivity,
        kinematic_viscosity=diffusivity,
        scalar_kind="mass",
        max_grid_peclet=2.0,
        max_schmidt=1000.0,
    )

    assert torch_audit["support_status"] == numpy_audit["support_status"]
    assert torch_audit["supported_accuracy"] == numpy_audit["supported_accuracy"]
    assert float(torch_audit["grid_peclet"]["max"]) == pytest.approx(
        float(numpy_audit["grid_peclet"]["max"]),
        rel=1e-12,
        abs=1e-12,
    )


def test_nonfinite_face_velocity_is_a_blocking_error_even_on_boundary() -> None:
    mesh = _one_cell_mesh()
    face_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    face_velocity[int(mesh.outlet_faces[0])] = np.nan

    audit = build_scalar_regime_audit(
        mesh,
        face_velocity,
        diffusivity=1e-6,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert audit["policy_version"] == "scalar_regime_guardrail_v2"
    assert audit["blocking_error"] is True
    assert audit["severity"] == "error"
    assert "invalid_nonfinite_face_normal_velocity" in audit["error_codes"]
    assert audit["diagnostics"]["nonfinite_face_velocity_count"] == 1


def test_infinite_face_velocity_is_a_blocking_error() -> None:
    mesh = _one_cell_mesh()
    audit = build_scalar_regime_audit(
        mesh,
        np.full(mesh.face_vertices.shape[0], np.inf, dtype=np.float64),
        diffusivity=1e-6,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert audit["blocking_error"] is True
    assert "invalid_nonfinite_face_normal_velocity" in audit["error_codes"]


def test_invalid_volume_and_nonfinite_grid_peclet_are_blocking_errors() -> None:
    mesh = _one_cell_mesh()
    mesh.cell_volumes[0] = np.nan
    audit = build_scalar_regime_audit(
        mesh,
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
        diffusivity=1e-6,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert audit["blocking_error"] is True
    assert "invalid_cell_volume" in audit["error_codes"]
    assert "invalid_nonfinite_grid_peclet" in audit["error_codes"]


def test_zero_volume_is_a_blocking_error() -> None:
    mesh = _one_cell_mesh()
    mesh.cell_volumes[0] = 0.0
    audit = build_scalar_regime_audit(
        mesh,
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
        diffusivity=1e-6,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert audit["blocking_error"] is True
    assert "invalid_cell_volume" in audit["error_codes"]


def test_nonfinite_face_area_is_a_blocking_error() -> None:
    mesh = _one_cell_mesh()
    mesh.face_areas[int(mesh.outlet_faces[0])] = np.nan
    audit = build_scalar_regime_audit(
        mesh,
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
        diffusivity=1e-6,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert audit["blocking_error"] is True
    assert "invalid_face_area" in audit["error_codes"]
    assert audit["diagnostics"]["nonfinite_face_area_count"] == 1


def test_incoming_boundary_flux_does_not_increase_grid_peclet() -> None:
    mesh = _one_cell_mesh()
    face_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    outlet_face = int(mesh.outlet_faces[0])
    face_velocity[outlet_face] = -1.0
    audit = build_scalar_regime_audit(
        mesh,
        face_velocity,
        diffusivity=1e-6,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert audit["grid_peclet"]["active_cell_count"] == 0
    assert audit["grid_peclet"]["max"] == pytest.approx(0.0)


def test_default_transport_pair_is_warning_only() -> None:
    mesh = _one_cell_mesh()
    audit = build_scalar_regime_audit(
        mesh,
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
        diffusivity=3e-10,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert "warning_high_schmidt" in audit["warning_codes"]
    assert audit["supported_accuracy"] is False
    assert audit["blocking_error"] is False
    assert audit["severity"] == "warning"


def test_zero_diffusivity_is_warning_only_advection_case() -> None:
    mesh = _one_cell_mesh()
    audit = build_scalar_regime_audit(
        mesh,
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
        diffusivity=0.0,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert audit["blocking_error"] is False
    assert audit["severity"] == "warning"
    assert "warning_zero_diffusivity_advection_only" in audit["warning_codes"]
    assert audit["inputs"]["dimensionless_ratio"] is None


@pytest.mark.parametrize("diffusivity", [-1e-6, float("nan"), float("inf")])
def test_invalid_diffusivity_is_blocking(diffusivity: float) -> None:
    mesh = _one_cell_mesh()
    audit = build_scalar_regime_audit(
        mesh,
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
        diffusivity=diffusivity,
        kinematic_viscosity=1e-6,
        scalar_kind="mass",
    )

    assert audit["blocking_error"] is True
    assert "invalid_diffusivity" in audit["error_codes"]
