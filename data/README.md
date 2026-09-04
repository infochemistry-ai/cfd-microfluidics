# Data

Only small, reproducible inputs needed by tests or documented examples belong
here. Generated mesh variants and calculation outputs are not tracked.

## Layout

- `meshes/gmsh/`: canonical `.msh` inputs
- `geometry/gmsh/`: canonical `.geo` sources
- `examples/`: small reproducible examples
- `examples/chemistry/`: versioned standalone chemistry mechanisms
- `examples/reactive/`: strict reactive-case JSON examples

## Rules

- New tracked meshes belong under `data/meshes/gmsh/`.
- New tracked geometry sources belong under `data/geometry/gmsh/`.
- Keep text `.geo` sources in normal Git when they are test fixtures.
- Keep tracked `.msh` fixtures small enough for ordinary Git.
- Every tracked input must have a test or documented example consumer.
- Imported inputs should record their source name, SHA-256 checksum, size,
  format, and generator version where applicable.
