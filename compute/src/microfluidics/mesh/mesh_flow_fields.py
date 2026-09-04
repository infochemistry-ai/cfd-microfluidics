"""Field data structures for mesh-native incompressible flow prototypes.

Current storage layout:
- velocity: cell-centered vector (n_cells, 3)
- pressure: cell-centered scalar (n_cells,)

This layout is chosen for consistency with the existing mesh scaffold and
operator base. A future face-centered (staggered-like) layout can be added
without breaking this API by introducing adapter utilities.
"""

from dataclasses import dataclass

import numpy as np

from microfluidics.mesh.mesh_builder import MeshDataModel
from microfluidics.mesh.mesh_fields import MeshScalarField, MeshVectorField


@dataclass
class MeshFlowFields:
    """Bundle of velocity and pressure fields on a mesh."""

    velocity: MeshVectorField
    pressure: MeshScalarField

    @property
    def mesh(self) -> MeshDataModel:
        return self.velocity.mesh

    def copy(self) -> "MeshFlowFields":
        return MeshFlowFields(
            velocity=self.velocity.copy(name=self.velocity.name),
            pressure=self.pressure.copy(name=self.pressure.name),
        )


def zeros_flow_fields(mesh: MeshDataModel) -> MeshFlowFields:
    return MeshFlowFields(
        velocity=MeshVectorField(
            mesh=mesh,
            values=np.zeros((mesh.cells.shape[0], 3), dtype=np.float64),
            name="velocity_native",
        ),
        pressure=MeshScalarField(
            mesh=mesh,
            values=np.zeros(mesh.cells.shape[0], dtype=np.float64),
            name="pressure_native",
        ),
    )
