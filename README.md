# Microfluidics CFD

This repository contains a standalone microfluidics computation surface. It
imports tetrahedral Gmsh meshes and runs the built-in finite-volume
flow, scalar transport, thermal transport, and reactive transport stages. The
same real stage entrypoints are available through a manifest-first CLI, a
local HTTP API, and an MCP server.

The repository is self-contained.

## What is included

- `compute/`: mesh import, preprocessing, numerical kernels, chemistry, and
  CPU/CUDA implementations.
- `experiments/gmsh/`: real import, flow, transport, thermal, and reactive
  entrypoints plus the local pipeline runner.
- `backend/`: local subprocess execution and the optional HTTP API.
- `mcp-server/`: local-only MCP tools for meshes, stages, status, artifacts,
  and cancellation.
- `shared/contracts/`: request, response, and runtime configuration models.
- `data/`: two small runnable meshes, two Gmsh geometry sources, and compact
  example case files.
- `tools/`: environment checking, CPU scaling, and Poiseuille verification.

## Platform and hardware

- Python: CPython 3.11 or 3.12, 64-bit.
- Operating systems: Windows x86-64 is tested. Linux x86-64 is supported by
  the locked dependencies but should be validated on the target machine.
  macOS is not supported by the frozen CPU-wheel configuration.
- CPU: x86-64 CPU; four or more cores are recommended. Set
  `MICROFLUIDICS_CPU_THREADS` and the usual BLAS/OpenMP variables to cap
  threads.
- RAM: 8 GB is a practical starting point for the included examples. Memory
  grows with mesh cells and enabled fields; no fixed upper bound is claimed.
- Disk: allow at least 5 GB for the environment and example results. Long
  transient runs can require substantially more.
- GPU: optional. Flow, scalar transport, and thermal stages can use CUDA.
  Reactive transport v1 is CPU-only. GPU memory requirements have not been
  benchmarked and depend on mesh size.

The numerical CFD solver is included; no additional CFD solver is required.
An external Gmsh executable is optional and is
needed only to generate a new mesh from `.geo`, BRep, or STEP input. Existing
`.msh` files run without it. The importer accepts the Gmsh MSH 4.1 format.
The exact external Gmsh release is not pinned.

Procedural CAD additionally uses `pythonocc-core==7.9.3` from conda-forge;
see `compute/environment.cad.yml`. This optional CAD path is outside the
default test suite.

## Install the CPU environment

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run from
its root:

```bash
uv sync --frozen --all-packages --group dev
```

This creates `.venv` and installs every workspace package. The frozen default
uses PyTorch 2.5.1 CPU wheels.

## Install CUDA support

[PyTorch 2.5.1](https://docs.pytorch.org/get-started/previous-versions/)
publishes CUDA 11.8, 12.1, and 12.4 wheels for Windows and Linux. The
following replaces the CPU wheel with the CUDA 12.4 build while keeping the
project's PyTorch version:

Windows PowerShell:

```powershell
uv pip install --python .venv\Scripts\python.exe --reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Linux:

```bash
uv pip install --python .venv/bin/python --reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

After replacing the wheel, keep `--no-sync` on `uv run` commands; a normal
`uv sync` restores the locked CPU wheel. For CUDA 12.4, use a compatible
NVIDIA GPU and driver. The
[CUDA 12.4 release notes](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/)
give a conservative GA baseline of driver 550.54.14 or newer on Linux and
551.61 or newer on Windows.

Select the visible GPU with `CUDA_VISIBLE_DEVICES`. Direct CLIs accept a
device such as `cuda:0`; MCP and HTTP stage requests use `"device":"cuda"`
and select the first visible device. No GPU number is hard-coded.

## Configuration

Direct CLI runs use CLI arguments. The HTTP and MCP services use environment
variables. `.env.example` lists safe local defaults but is not loaded
automatically. At minimum, enable execution before starting a service:

Windows PowerShell:

```powershell
$env:SERVICE_ENABLED = "true"
```

Linux:

```bash
export SERVICE_ENABLED=true
```

The default listeners bind only to `127.0.0.1`. If a listener is exposed to
another machine, set a strong `SERVICE_API_KEY` and protect transport at the
network boundary. Stage inputs must be repository-relative POSIX paths and
must resolve inside the checkout. Results default to
`results/service_runs/`.

## Check the environment

CPU:

```bash
uv run --no-sync python tools/check_environment.py --device cpu
```

CUDA:

```bash
uv run --no-sync python tools/check_environment.py --device cuda
```

To require mesh-generation support too:

```bash
uv run --no-sync python tools/check_environment.py --require-gmsh
```

The command prints JSON and exits zero only when all requested capabilities are
available. A missing GPU or Gmsh produces a specific non-zero failure instead
of silently changing execution mode.

## Minimal real CFD run

This fast smoke follows the real Gmsh import and pressure-projection flow path
on the included vertical pipe mesh:

```bash
uv run --no-sync python experiments/gmsh/run_gmsh_pipeline_manifest.py --run-root results/quickstart --msh data/meshes/gmsh/vertical_pipe_500.msh --mesh-name vertical_pipe_500 --flow-backend numpy --flow-mode projection_only --flow-steps 1 --skip-transport
```

Success means the command exits zero, `results/quickstart/pipeline_manifest.json`
exists, and the manifest contains completed import and flow runs. Each stage
also writes a timestamped directory containing `summary.json`, `run.log`,
NumPy arrays, and preview images where applicable.

## Longer physical flow verification

The Poiseuille harness runs the production no-slip solver and compares pressure
gradient, velocity profile, conservation, and stationarity against a circular
pipe reference:

```bash
uv run --no-sync python tools/run_vertical_pipe_poiseuille_verification.py --mesh data/meshes/gmsh/vertical_pipe_500.msh --steps 700 --min-steps 500 --stop-when-steady --pressure-relative-tolerance 1e-7 --require-acceptance --output-dir results/poiseuille
```

This is much slower than the quickstart. With `--require-acceptance`, a
non-zero exit means at least one physical acceptance gate failed. The report is
written to `results/poiseuille/report.json` and `report.md`.

For CUDA flow on a direct pipeline, replace the flow arguments with:

```text
--flow-backend torch --flow-device cuda:0
```

Transport and thermal have independent device arguments:
`--transport-execution-backend torch --transport-torch-device cuda:0` and
`--thermal-backend torch --thermal-torch-device cuda:0`.

## HTTP API

Start the local API:

```bash
SERVICE_ENABLED=true uv run --no-sync python backend/app/app.py
```

On PowerShell, set `$env:SERVICE_ENABLED="true"` first and then run the Python
command. Submit an import:

```bash
curl -X POST http://127.0.0.1:8091/api/v1/compute -H "Content-Type: application/json" -d '{"contract_version":"v1","request_id":"import-001","experiment_id":"stage_import","parameters":{"mesh_path":"data/meshes/gmsh/vertical_pipe_500.msh"}}'
```

The HTTP endpoint is synchronous and returns a terminal result. Copy the
returned imported `.npz` path into a `stage_flow` request. The other allowed
stage IDs are `stage_transport`, `stage_thermal`, and
`stage_reactive_transport`. OpenAPI is available at
`http://127.0.0.1:8091/openapi.json`.

To request cancellation from another client while a synchronous request is
running:

```bash
curl -X POST http://127.0.0.1:8091/api/v1/compute/import-001/cancel
```

Stop the service with Ctrl+C.

## MCP server

For stdio MCP:

```bash
SERVICE_ENABLED=true uv run --no-sync python -m microfluidics_mcp
```

Configure an MCP client to launch that command from the repository root. The
tools are:

- mesh discovery and registration: `cfd_list_meshes`,
  `cfd_register_local_mesh`;
- case validation: `cfd_validate_reactive_case`;
- five stage launchers: `cfd_run_import`, `cfd_run_flow`,
  `cfd_run_transport`, `cfd_run_thermal`,
  `cfd_run_reactive_transport`;
- status, artifacts, and cancellation: `cfd_get_run`,
  `cfd_list_artifacts`, `cfd_get_artifact`, `cfd_cancel_run`.

Stage tools return immediately. Poll `cfd_get_run`, then copy its `outputs`
paths verbatim into the next stage. All computation remains on the local
machine.

For local streamable HTTP MCP, set `MCP_ENABLED=true` and start
`backend/app/app.py`; the MCP endpoint defaults to
`http://127.0.0.1:8092/mcp`.

## Inputs and outputs

Inputs:

- Gmsh MSH 4.1 tetrahedral mesh with physical inlet, outlet, wall, and fluid
  groups;
- optional `case_config_v1` JSON from `data/examples/cfd/`;
- optional reactive-case JSON from `data/examples/reactive/`;
- optional Gmsh `.geo` or CAD JSON when generating geometry.

The import stage creates a portable `.npz` mesh representation. Flow creates
`summary.json`, `flow_coupling_metadata.json`,
`final_corrected_face_flux.npy`, `face_to_cells.npy`, and
`cell_volumes.npy`. Downstream stages consume these exact files. Results are
under the requested `--run-root` for direct pipelines or
`SERVICE_RUN_ROOT` for API/MCP runs. `results/` is ignored by Git.

## Tests

Fast validation suite:

```bash
uv run --no-sync pytest -p no:cacheprovider -q -m "not slow and not performance and not system"
uv run --no-sync ruff check .
```

Optional external-CAD/Gmsh tests skip when their executables are unavailable.
CUDA tests skip without a usable GPU. Performance, slow, and system tests are
not part of the default validation suite.

## Troubleshooting

- `CUDA requested, but no usable CUDA device is visible`: confirm the CUDA
  PyTorch wheel, NVIDIA driver, and `CUDA_VISIBLE_DEVICES`; rerun the CUDA
  environment check.
- `Gmsh ... not found`: install Gmsh and add it to `PATH`, or use an
  existing `.msh` file.
- input outside project root: copy the input into `data/` and pass a
  repository-relative path with forward slashes.
- stage reports `failed`: inspect its `compute.log` or `run.log` and
  `summary.json`.
- dependency drift: rerun `uv sync --frozen --all-packages --group dev`.
- port already in use: change `SERVICE_PORT` or `MCP_PORT`.

## Known limitations

- CUDA execution requires a compatible NVIDIA GPU and driver; hardware tests
  skip automatically when CUDA is unavailable.
- External Gmsh and the OpenCASCADE CAD environment are optional and are not
  covered by the default test suite.
- Reactive transport v1 has no CUDA implementation.
- Runtime records are in memory and are lost when the API/MCP process exits;
  artifacts remain on disk.

## License

This software, including its bundled example meshes, geometry, and
configuration data, is licensed under the MIT License. See `LICENSE`.
