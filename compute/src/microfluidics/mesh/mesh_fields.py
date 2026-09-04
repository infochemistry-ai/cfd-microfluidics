"""Field data structures for mesh-native numerical prototypes."""

from dataclasses import dataclass

import numpy as np

from microfluidics.mesh.mesh_builder import MeshDataModel


@dataclass
class MeshScalarField:
    """Cell-centered scalar field on mesh cells."""

    mesh: MeshDataModel
    values: np.ndarray
    name: str = "scalar"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if self.values.shape[0] != self.mesh.cells.shape[0]:
            raise ValueError(
                f"{self.name} field size mismatch: {self.values.shape[0]} values for "
                f"{self.mesh.cells.shape[0]} mesh cells."
            )

    def copy(self, name: str | None = None) -> "MeshScalarField":
        return MeshScalarField(self.mesh, self.values.copy(), name=name or self.name)


@dataclass
class MeshVectorField:
    """Cell-centered vector field with 3 components."""

    mesh: MeshDataModel
    values: np.ndarray
    name: str = "vector"

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{self.name} must be shaped (n_cells, 3).")
        if arr.shape[0] != self.mesh.cells.shape[0]:
            raise ValueError(
                f"{self.name} field size mismatch: {arr.shape[0]} values for "
                f"{self.mesh.cells.shape[0]} mesh cells."
            )
        self.values = arr

    def copy(self, name: str | None = None) -> "MeshVectorField":
        return MeshVectorField(self.mesh, self.values.copy(), name=name or self.name)


def zeros_scalar_field(mesh: MeshDataModel, name: str = "scalar") -> MeshScalarField:
    return MeshScalarField(
        mesh, np.zeros(mesh.cells.shape[0], dtype=np.float64), name=name
    )


def constant_scalar_field(
    mesh: MeshDataModel,
    value: float,
    name: str = "scalar",
) -> MeshScalarField:
    return MeshScalarField(
        mesh,
        np.full(mesh.cells.shape[0], float(value), dtype=np.float64),
        name=name,
    )
