"""Build the real solver command line for each local CFD stage."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from backend.app.stage_registry import (
    STAGE_REGISTRY,
    Device,
    FlowNumericalProfile,
    StageDefinition,
    StageFlowParameters,
    StageImportParameters,
    StageParameters,
    StageReactiveTransportParameters,
    StageThermalParameters,
    StageTransportParameters,
)

STAGE_SCRIPTS: Mapping[str, str] = MappingProxyType(
    {
        "stage_import": "experiments/gmsh/run_import_gmsh_mesh.py",
        "stage_flow": "experiments/gmsh/run_gmsh_tetra_flow_debug.py",
        "stage_transport": "experiments/gmsh/run_gmsh_tetra_transport_debug.py",
        "stage_thermal": "experiments/gmsh/run_gmsh_tetra_thermal_debug.py",
        "stage_reactive_transport": "experiments/gmsh/run_gmsh_tetra_reactive_debug.py",
    }
)


@dataclass(frozen=True)
class StageLayout:
    """Where the local executor keeps sources, staged inputs and outputs."""

    python: str
    app_root: Path
    work_root: Path
    output_root: Path

    def script(self, relative_posix: str) -> str:
        """Resolve a repository-relative entrypoint against this layout."""

        return str(self.app_root / Path(relative_posix))

    def staged(self, staged_path: str) -> str:
        """Resolve a registry staged path inside the local work directory."""

        return str(self.work_root / Path(staged_path))

    def staged_dir(self, staged_path: str) -> str:
        """The directory a registry staged path sits in, in this layout."""

        return str((self.work_root / Path(staged_path)).parent)


def local_stage_layout(
    *,
    project_root: Path,
    work_root: Path,
    output_root: Path,
    python: str | None = None,
) -> StageLayout:
    """The layout of a checkout, with staging and results inside one run dir.

    `python` defaults to the interpreter running the service, which is the
    interpreter that has this repository's dependencies installed.
    """

    return StageLayout(
        python=python or sys.executable,
        app_root=project_root,
        work_root=work_root,
        output_root=output_root,
    )


def staged_input(
    definition: StageDefinition,
    parameter_name: str,
    layout: StageLayout,
) -> str:
    """Where the file behind `parameter_name` waits for the solver."""

    for binding in definition.inputs:
        if binding.parameter_name == parameter_name:
            return layout.staged(binding.staged_path)
    raise KeyError(
        f"Stage {definition.experiment_id.value!r} has no input "
        f"parameter {parameter_name!r}."
    )


def staged_input_dir(
    definition: StageDefinition,
    parameter_name: str,
    layout: StageLayout,
) -> str:
    """The directory the file behind `parameter_name` is staged into."""

    for binding in definition.inputs:
        if binding.parameter_name == parameter_name:
            return layout.staged_dir(binding.staged_path)
    raise KeyError(
        f"Stage {definition.experiment_id.value!r} has no input "
        f"parameter {parameter_name!r}."
    )


def _import_argv(
    definition: StageDefinition,
    parameters: StageImportParameters,
    layout: StageLayout,
) -> list[str]:
    _ = parameters
    return [
        layout.python,
        layout.script(STAGE_SCRIPTS["stage_import"]),
        "--msh",
        staged_input(definition, "mesh_path", layout),
        "--output-root",
        str(layout.output_root),
    ]


def _flow_argv(
    definition: StageDefinition,
    parameters: StageFlowParameters,
    layout: StageLayout,
) -> list[str]:
    argv = [
        layout.python,
        layout.script(STAGE_SCRIPTS["stage_flow"]),
        "--mesh-npz",
        staged_input(definition, "mesh_npz_path", layout),
        "--output-root",
        str(layout.output_root),
        "--flow-steps",
        str(parameters.num_steps),
        "--device",
        parameters.device.value,
    ]
    if parameters.device is Device.CUDA:
        argv.extend(
            [
                "--backend",
                "torch",
                "--flow-execution-backend",
                "torch",
                "--fail-if-numpy-fallback",
            ]
        )
    if parameters.flow_stop_physical_time is not None:
        argv.extend(
            ["--flow-stop-physical-time", str(parameters.flow_stop_physical_time)]
        )
    if parameters.snapshot_time_interval is not None:
        argv.extend(
            ["--snapshot-time-interval", str(parameters.snapshot_time_interval)]
        )
    if (
        parameters.numerical_profile
        is FlowNumericalProfile.NO_SLIP_TJUNCTION_VALIDATION_V1
    ):
        argv.extend(
            [
                "--wall-velocity-boundary-mode",
                "no_slip",
                "--flow-dt-mode",
                "auto_cfl",
                "--convective-cfl-target",
                "0.45",
                "--flow-mode",
                "navier_stokes_projection_debug",
                "--disable-convective-auto-damping",
                "--convective-stabilization-mode",
                "auto_damping",
                "--viscous-face-flux-divergence-impact-cap",
                "0.03",
                "--pressure-nonorthogonal-correction-sweeps",
                "4",
                "--pressure-nonorthogonal-correction-relaxation",
                "1.0",
                "--pressure-solver",
                "pcg_diag",
                "--max-pressure-iterations",
                "1000",
                "--pressure-relative-tolerance",
                "1e-6",
                "--pcg-require-relative-l2-convergence",
                "--startup-warning-steps",
                "10",
            ]
        )
    return argv


def _transport_argv(
    definition: StageDefinition,
    parameters: StageTransportParameters,
    layout: StageLayout,
) -> list[str]:
    argv = [
        layout.python,
        layout.script(STAGE_SCRIPTS["stage_transport"]),
        "--mesh-npz",
        staged_input(definition, "mesh_npz_path", layout),
        "--output-root",
        str(layout.output_root),
        "--velocity-source",
        "flow_run",
        "--flow-run-dir",
        staged_input_dir(definition, "flow_coupling_metadata_path", layout),
        "--steps",
        str(parameters.num_steps),
        "--transport-execution-backend",
        "torch",
        "--torch-device",
        parameters.device.value,
    ]
    # Same condition as the flow stage: the entrypoint reads this flag as an
    # assertion that stepping ran on CUDA, so on CPU it can only ever fail the
    # run after the solver has already done the work.
    if parameters.device is Device.CUDA:
        argv.append("--fail-if-numpy-fallback")
    return argv


def _thermal_argv(
    definition: StageDefinition,
    parameters: StageThermalParameters,
    layout: StageLayout,
) -> list[str]:
    return [
        layout.python,
        layout.script(STAGE_SCRIPTS["stage_thermal"]),
        "--msh",
        staged_input(definition, "mesh_path", layout),
        "--output-root",
        str(layout.output_root),
        "--velocity-source",
        "flow_solver",
        "--flow-summary-json",
        staged_input(definition, "flow_summary_path", layout),
        "--steps",
        str(parameters.num_steps),
        "--backend",
        "torch",
        "--torch-device",
        parameters.device.value,
    ]


def _reactive_transport_argv(
    definition: StageDefinition,
    parameters: StageReactiveTransportParameters,
    layout: StageLayout,
) -> list[str]:
    argv = [
        layout.python,
        layout.script(STAGE_SCRIPTS["stage_reactive_transport"]),
        "--mesh-npz",
        staged_input(definition, "mesh_npz_path", layout),
        "--flow-run-dir",
        staged_input_dir(definition, "flow_summary_path", layout),
        "--reactive-case",
        staged_input(definition, "reactive_case_path", layout),
        "--output-root",
        str(layout.output_root),
    ]
    if parameters.max_walltime_seconds > 0.0:
        argv.extend(
            [
                "--max-walltime-seconds",
                str(parameters.max_walltime_seconds),
            ]
        )
    return argv


_ARGV_BUILDERS: Mapping[
    str, Callable[[StageDefinition, StageParameters, StageLayout], list[str]]
] = MappingProxyType(
    {
        "stage_import": _import_argv,  # type: ignore[dict-item]
        "stage_flow": _flow_argv,  # type: ignore[dict-item]
        "stage_transport": _transport_argv,  # type: ignore[dict-item]
        "stage_thermal": _thermal_argv,  # type: ignore[dict-item]
        "stage_reactive_transport": _reactive_transport_argv,  # type: ignore[dict-item]
    }
)


def build_stage_solver_argv(
    experiment_id: str,
    parameters: StageParameters,
    layout: StageLayout,
) -> list[str]:
    """The solver invocation for one stage, expressed in `layout`."""

    try:
        builder = _ARGV_BUILDERS[experiment_id]
        definition = STAGE_REGISTRY[experiment_id]
    except KeyError as exc:
        raise ValueError(f"Unknown CFD stage {experiment_id!r}.") from exc
    return builder(definition, parameters, layout)
