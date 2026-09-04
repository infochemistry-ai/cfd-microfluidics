"""Strict public contract and immutable registry for runnable CFD stages."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence, TypeAlias

from microfluidics_contracts import RuntimeSettings


class StageId(str, Enum):
    IMPORT = "stage_import"
    FLOW = "stage_flow"
    TRANSPORT = "stage_transport"
    THERMAL = "stage_thermal"
    REACTIVE_TRANSPORT = "stage_reactive_transport"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class FlowNumericalProfile(str, Enum):
    DEFAULT = "default"
    NO_SLIP_TJUNCTION_VALIDATION_V1 = "no_slip_tjunction_validation_v1"


class StageParametersError(ValueError):
    """A client supplied an invalid stage-specific parameter object."""


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_PATH_SEGMENT_PATTERN = r"(?!\.{1,2}(?:/|$))[^/\\\x00-\x1f\x7f]+"

# Accepted input-prefix type for project-root-scoped stage paths.
InputPrefixes: TypeAlias = "str | Sequence[str]"


def normalize_input_prefixes(value: InputPrefixes) -> tuple[str, ...]:
    """Order-preserving, de-duplicated, slash-trimmed, empties dropped."""

    items = (value,) if isinstance(value, str) else tuple(value)
    prefixes: list[str] = []
    for item in items:
        prefix = str(item).strip().strip("/")
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    return tuple(prefixes)


def input_path_schema_pattern(*, suffix: str, input_prefixes: InputPrefixes = "") -> str:
    """Return a JSON Schema pattern for safe repository-relative paths."""

    prefixes = normalize_input_prefixes(input_prefixes)
    alternatives = "|".join(re.escape(prefix) + "/" for prefix in prefixes)
    if not prefixes:
        required_prefix = ""
    elif len(prefixes) == 1:
        required_prefix = alternatives
    else:
        required_prefix = f"(?:{alternatives})"
    return (
        rf"^(?=.+{re.escape(suffix)}$)"
        rf"{required_prefix}{_PATH_SEGMENT_PATTERN}"
        rf"(?:/{_PATH_SEGMENT_PATTERN})*$"
    )


def validate_input_path(
    value: object,
    *,
    field_name: str,
    suffix: str,
    input_prefixes: InputPrefixes = "",
) -> str:
    if not isinstance(value, str):
        raise StageParametersError(f"'{field_name}' must be a string.")
    path = value
    if not (1 <= len(path) <= 1024):
        raise StageParametersError(f"'{field_name}' length must be between 1 and 1024.")
    if _URI_SCHEME.match(path):
        raise StageParametersError(
            f"'{field_name}' must be a repository-relative path, not a URL."
        )
    if path.startswith("/") or "\\" in path or _CONTROL_CHARS.search(path):
        raise StageParametersError(f"'{field_name}' is not a safe relative POSIX path.")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise StageParametersError(f"'{field_name}' contains an unsafe path segment.")
    if not path.endswith(suffix):
        raise StageParametersError(f"'{field_name}' must end with {suffix!r}.")
    prefixes = normalize_input_prefixes(input_prefixes)
    if prefixes and not any(path.startswith(prefix + "/") for prefix in prefixes):
        raise StageParametersError(
            f"'{field_name}' must be located under the configured input prefix."
        )
    return path


def stage_input_prefixes(settings: RuntimeSettings) -> tuple[str, ...]:
    """Standalone inputs are constrained by project-root containment."""

    _ = settings
    return ()


def _strict_num_steps(value: object) -> int:
    if type(value) is not int:  # bool is deliberately rejected
        raise StageParametersError("'num_steps' must be an integer.")
    if not 1 <= value <= 1_000_000:
        raise StageParametersError("'num_steps' must be between 1 and 1000000.")
    return value


def _strict_positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageParametersError(f"'{field_name}' must be a number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise StageParametersError(f"'{field_name}' must be finite and positive.")
    return parsed


def _nonnegative_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageParametersError("'max_walltime_seconds' must be a number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise StageParametersError(
            "'max_walltime_seconds' must be finite and non-negative."
        )
    return parsed


def _device(value: object) -> Device:
    if not isinstance(value, str):
        raise StageParametersError("'device' must be 'cpu' or 'cuda'.")
    try:
        return Device(value)
    except ValueError as exc:
        raise StageParametersError("'device' must be 'cpu' or 'cuda'.") from exc


def _flow_numerical_profile(value: object) -> FlowNumericalProfile:
    if not isinstance(value, str):
        raise StageParametersError(
            "'numerical_profile' must be 'default' or "
            "'no_slip_tjunction_validation_v1'."
        )
    try:
        return FlowNumericalProfile(value)
    except ValueError as exc:
        raise StageParametersError(
            "'numerical_profile' must be 'default' or "
            "'no_slip_tjunction_validation_v1'."
        ) from exc


def _cpu_device(value: object) -> Device:
    device = _device(value)
    if device is not Device.CPU:
        raise StageParametersError(
            "'device' must be 'cpu' for reactive transport v1; CUDA is not supported."
        )
    return device


@dataclass(frozen=True)
class StageParameters:
    allowed_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _validate_object(cls, raw: object) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise StageParametersError("'parameters' must be an object.")
        unknown = sorted(set(raw) - set(cls.allowed_fields))
        if unknown:
            raise StageParametersError(
                "Unsupported parameter(s): "
                + ", ".join(repr(item) for item in unknown)
                + "."
            )
        return raw

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class StageImportParameters(StageParameters):
    mesh_path: str
    allowed_fields: ClassVar[frozenset[str]] = frozenset({"mesh_path"})

    @classmethod
    def from_dict(
        cls, raw: object, *, input_prefixes: InputPrefixes = ""
    ) -> "StageImportParameters":
        values = cls._validate_object(raw)
        if "mesh_path" not in values:
            raise StageParametersError("'mesh_path' is required.")
        return cls(
            mesh_path=validate_input_path(
                values["mesh_path"],
                field_name="mesh_path",
                suffix=".msh",
                input_prefixes=input_prefixes,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"mesh_path": self.mesh_path}


@dataclass(frozen=True)
class StageFlowParameters(StageParameters):
    mesh_npz_path: str
    num_steps: int
    device: Device = Device.CPU
    numerical_profile: FlowNumericalProfile = FlowNumericalProfile.DEFAULT
    flow_stop_physical_time: float | None = None
    snapshot_time_interval: float | None = None
    allowed_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "mesh_npz_path",
            "num_steps",
            "device",
            "numerical_profile",
            "flow_stop_physical_time",
            "snapshot_time_interval",
        }
    )

    @classmethod
    def from_dict(
        cls, raw: object, *, input_prefixes: InputPrefixes = ""
    ) -> "StageFlowParameters":
        values = cls._validate_object(raw)
        for required in ("mesh_npz_path", "num_steps"):
            if required not in values:
                raise StageParametersError(f"'{required}' is required.")
        parsed = cls(
            mesh_npz_path=validate_input_path(
                values["mesh_npz_path"],
                field_name="mesh_npz_path",
                suffix=".npz",
                input_prefixes=input_prefixes,
            ),
            num_steps=_strict_num_steps(values["num_steps"]),
            device=_device(values.get("device", "cpu")),
            numerical_profile=_flow_numerical_profile(
                values.get("numerical_profile", FlowNumericalProfile.DEFAULT.value)
            ),
            flow_stop_physical_time=(
                _strict_positive_float(
                    values["flow_stop_physical_time"],
                    field_name="flow_stop_physical_time",
                )
                if values.get("flow_stop_physical_time") is not None
                else None
            ),
            snapshot_time_interval=(
                _strict_positive_float(
                    values["snapshot_time_interval"],
                    field_name="snapshot_time_interval",
                )
                if values.get("snapshot_time_interval") is not None
                else None
            ),
        )
        if (
            parsed.numerical_profile
            is FlowNumericalProfile.NO_SLIP_TJUNCTION_VALIDATION_V1
            and (
                parsed.flow_stop_physical_time is None
                or parsed.snapshot_time_interval is None
            )
        ):
            raise StageParametersError(
                "'no_slip_tjunction_validation_v1' requires both "
                "'flow_stop_physical_time' and 'snapshot_time_interval'."
            )
        if (
            parsed.flow_stop_physical_time is not None
            and parsed.snapshot_time_interval is not None
            and parsed.snapshot_time_interval > parsed.flow_stop_physical_time
        ):
            raise StageParametersError(
                "'snapshot_time_interval' must not exceed 'flow_stop_physical_time'."
            )
        return parsed

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mesh_npz_path": self.mesh_npz_path,
            "num_steps": self.num_steps,
            "device": self.device.value,
            "numerical_profile": self.numerical_profile.value,
        }
        if self.flow_stop_physical_time is not None:
            result["flow_stop_physical_time"] = self.flow_stop_physical_time
        if self.snapshot_time_interval is not None:
            result["snapshot_time_interval"] = self.snapshot_time_interval
        return result


@dataclass(frozen=True)
class StageTransportParameters(StageParameters):
    mesh_npz_path: str
    flow_coupling_metadata_path: str
    flow_face_flux_path: str
    num_steps: int
    device: Device = Device.CPU
    allowed_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "mesh_npz_path",
            "flow_coupling_metadata_path",
            "flow_face_flux_path",
            "num_steps",
            "device",
        }
    )

    @classmethod
    def from_dict(
        cls, raw: object, *, input_prefixes: InputPrefixes = ""
    ) -> "StageTransportParameters":
        values = cls._validate_object(raw)
        required = (
            "mesh_npz_path",
            "flow_coupling_metadata_path",
            "flow_face_flux_path",
            "num_steps",
        )
        for name in required:
            if name not in values:
                raise StageParametersError(f"'{name}' is required.")
        return cls(
            mesh_npz_path=validate_input_path(
                values["mesh_npz_path"],
                field_name="mesh_npz_path",
                suffix=".npz",
                input_prefixes=input_prefixes,
            ),
            flow_coupling_metadata_path=validate_input_path(
                values["flow_coupling_metadata_path"],
                field_name="flow_coupling_metadata_path",
                suffix=".json",
                input_prefixes=input_prefixes,
            ),
            flow_face_flux_path=validate_input_path(
                values["flow_face_flux_path"],
                field_name="flow_face_flux_path",
                suffix=".npy",
                input_prefixes=input_prefixes,
            ),
            num_steps=_strict_num_steps(values["num_steps"]),
            device=_device(values.get("device", "cpu")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_npz_path": self.mesh_npz_path,
            "flow_coupling_metadata_path": self.flow_coupling_metadata_path,
            "flow_face_flux_path": self.flow_face_flux_path,
            "num_steps": self.num_steps,
            "device": self.device.value,
        }


@dataclass(frozen=True)
class StageThermalParameters(StageParameters):
    mesh_path: str
    flow_summary_path: str
    flow_coupling_metadata_path: str
    flow_face_flux_path: str
    flow_face_to_cells_path: str
    flow_cell_volumes_path: str
    num_steps: int
    device: Device = Device.CPU
    allowed_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "mesh_path",
            "flow_summary_path",
            "flow_coupling_metadata_path",
            "flow_face_flux_path",
            "flow_face_to_cells_path",
            "flow_cell_volumes_path",
            "num_steps",
            "device",
        }
    )

    @classmethod
    def from_dict(
        cls, raw: object, *, input_prefixes: InputPrefixes = ""
    ) -> "StageThermalParameters":
        values = cls._validate_object(raw)
        required = (
            "mesh_path",
            "flow_summary_path",
            "flow_coupling_metadata_path",
            "flow_face_flux_path",
            "flow_face_to_cells_path",
            "flow_cell_volumes_path",
            "num_steps",
        )
        for name in required:
            if name not in values:
                raise StageParametersError(f"'{name}' is required.")
        return cls(
            mesh_path=validate_input_path(
                values["mesh_path"],
                field_name="mesh_path",
                suffix=".msh",
                input_prefixes=input_prefixes,
            ),
            flow_summary_path=validate_input_path(
                values["flow_summary_path"],
                field_name="flow_summary_path",
                suffix=".json",
                input_prefixes=input_prefixes,
            ),
            flow_coupling_metadata_path=validate_input_path(
                values["flow_coupling_metadata_path"],
                field_name="flow_coupling_metadata_path",
                suffix=".json",
                input_prefixes=input_prefixes,
            ),
            flow_face_flux_path=validate_input_path(
                values["flow_face_flux_path"],
                field_name="flow_face_flux_path",
                suffix=".npy",
                input_prefixes=input_prefixes,
            ),
            flow_face_to_cells_path=validate_input_path(
                values["flow_face_to_cells_path"],
                field_name="flow_face_to_cells_path",
                suffix=".npy",
                input_prefixes=input_prefixes,
            ),
            flow_cell_volumes_path=validate_input_path(
                values["flow_cell_volumes_path"],
                field_name="flow_cell_volumes_path",
                suffix=".npy",
                input_prefixes=input_prefixes,
            ),
            num_steps=_strict_num_steps(values["num_steps"]),
            device=_device(values.get("device", "cpu")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_path": self.mesh_path,
            "flow_summary_path": self.flow_summary_path,
            "flow_coupling_metadata_path": self.flow_coupling_metadata_path,
            "flow_face_flux_path": self.flow_face_flux_path,
            "flow_face_to_cells_path": self.flow_face_to_cells_path,
            "flow_cell_volumes_path": self.flow_cell_volumes_path,
            "num_steps": self.num_steps,
            "device": self.device.value,
        }


@dataclass(frozen=True)
class StageReactiveTransportParameters(StageParameters):
    mesh_npz_path: str
    flow_summary_path: str
    flow_coupling_metadata_path: str
    flow_face_flux_path: str
    flow_face_to_cells_path: str
    flow_cell_volumes_path: str
    reactive_case_path: str
    device: Device = Device.CPU
    max_walltime_seconds: float = 0.0
    allowed_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "mesh_npz_path",
            "flow_summary_path",
            "flow_coupling_metadata_path",
            "flow_face_flux_path",
            "flow_face_to_cells_path",
            "flow_cell_volumes_path",
            "reactive_case_path",
            "device",
            "max_walltime_seconds",
        }
    )

    @classmethod
    def from_dict(
        cls, raw: object, *, input_prefixes: InputPrefixes = ""
    ) -> "StageReactiveTransportParameters":
        values = cls._validate_object(raw)
        required = (
            "mesh_npz_path",
            "flow_summary_path",
            "flow_coupling_metadata_path",
            "flow_face_flux_path",
            "flow_face_to_cells_path",
            "flow_cell_volumes_path",
            "reactive_case_path",
        )
        for name in required:
            if name not in values:
                raise StageParametersError(f"'{name}' is required.")
        return cls(
            mesh_npz_path=validate_input_path(
                values["mesh_npz_path"],
                field_name="mesh_npz_path",
                suffix=".npz",
                input_prefixes=input_prefixes,
            ),
            flow_summary_path=validate_input_path(
                values["flow_summary_path"],
                field_name="flow_summary_path",
                suffix=".json",
                input_prefixes=input_prefixes,
            ),
            flow_coupling_metadata_path=validate_input_path(
                values["flow_coupling_metadata_path"],
                field_name="flow_coupling_metadata_path",
                suffix=".json",
                input_prefixes=input_prefixes,
            ),
            flow_face_flux_path=validate_input_path(
                values["flow_face_flux_path"],
                field_name="flow_face_flux_path",
                suffix=".npy",
                input_prefixes=input_prefixes,
            ),
            flow_face_to_cells_path=validate_input_path(
                values["flow_face_to_cells_path"],
                field_name="flow_face_to_cells_path",
                suffix=".npy",
                input_prefixes=input_prefixes,
            ),
            flow_cell_volumes_path=validate_input_path(
                values["flow_cell_volumes_path"],
                field_name="flow_cell_volumes_path",
                suffix=".npy",
                input_prefixes=input_prefixes,
            ),
            reactive_case_path=validate_input_path(
                values["reactive_case_path"],
                field_name="reactive_case_path",
                suffix=".json",
                input_prefixes=input_prefixes,
            ),
            device=_cpu_device(values.get("device", "cpu")),
            max_walltime_seconds=_nonnegative_seconds(
                values.get("max_walltime_seconds", 0.0)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_npz_path": self.mesh_npz_path,
            "flow_summary_path": self.flow_summary_path,
            "flow_coupling_metadata_path": (self.flow_coupling_metadata_path),
            "flow_face_flux_path": self.flow_face_flux_path,
            "flow_face_to_cells_path": self.flow_face_to_cells_path,
            "flow_cell_volumes_path": self.flow_cell_volumes_path,
            "reactive_case_path": self.reactive_case_path,
            "device": self.device.value,
            "max_walltime_seconds": self.max_walltime_seconds,
        }


@dataclass(frozen=True)
class StageImportComputeRequest:
    request_id: str
    parameters: StageImportParameters
    experiment_id: ClassVar[StageId] = StageId.IMPORT


@dataclass(frozen=True)
class StageFlowComputeRequest:
    request_id: str
    parameters: StageFlowParameters
    experiment_id: ClassVar[StageId] = StageId.FLOW


@dataclass(frozen=True)
class StageTransportComputeRequest:
    request_id: str
    parameters: StageTransportParameters
    experiment_id: ClassVar[StageId] = StageId.TRANSPORT


@dataclass(frozen=True)
class StageThermalComputeRequest:
    request_id: str
    parameters: StageThermalParameters
    experiment_id: ClassVar[StageId] = StageId.THERMAL


@dataclass(frozen=True)
class StageReactiveTransportComputeRequest:
    request_id: str
    parameters: StageReactiveTransportParameters
    experiment_id: ClassVar[StageId] = StageId.REACTIVE_TRANSPORT


CFDComputeRequest: TypeAlias = (
    StageImportComputeRequest
    | StageFlowComputeRequest
    | StageTransportComputeRequest
    | StageThermalComputeRequest
    | StageReactiveTransportComputeRequest
)


@dataclass(frozen=True)
class StageInputBinding:
    """Where one stage parameter is copied inside the local run directory."""

    parameter_name: str
    required_suffix: str
    staged_path: str


@dataclass(frozen=True)
class StageDefinition:
    experiment_id: StageId
    parameters_model: type[StageParameters]
    inputs: tuple[StageInputBinding, ...]


STAGE_REGISTRY: Mapping[str, StageDefinition] = MappingProxyType(
    {
        StageId.IMPORT.value: StageDefinition(
            StageId.IMPORT,
            StageImportParameters,
            (
                StageInputBinding(
                    "mesh_path",
                    ".msh",
                    "input.msh",
                ),
            ),
        ),
        StageId.FLOW.value: StageDefinition(
            StageId.FLOW,
            StageFlowParameters,
            (
                StageInputBinding(
                    "mesh_npz_path",
                    ".npz",
                    "mesh.npz",
                ),
            ),
        ),
        StageId.TRANSPORT.value: StageDefinition(
            StageId.TRANSPORT,
            StageTransportParameters,
            (
                StageInputBinding(
                    "mesh_npz_path",
                    ".npz",
                    "mesh.npz",
                ),
                StageInputBinding(
                    "flow_coupling_metadata_path",
                    ".json",
                    "flow_run/flow_coupling_metadata.json",
                ),
                StageInputBinding(
                    "flow_face_flux_path",
                    ".npy",
                    "flow_run/final_corrected_face_flux.npy",
                ),
            ),
        ),
        StageId.THERMAL.value: StageDefinition(
            StageId.THERMAL,
            StageThermalParameters,
            (
                StageInputBinding(
                    "mesh_path",
                    ".msh",
                    "input.msh",
                ),
                StageInputBinding(
                    "flow_summary_path",
                    ".json",
                    "flow_run/summary.json",
                ),
                StageInputBinding(
                    "flow_coupling_metadata_path",
                    ".json",
                    "flow_run/flow_coupling_metadata.json",
                ),
                StageInputBinding(
                    "flow_face_flux_path",
                    ".npy",
                    "flow_run/final_corrected_face_flux.npy",
                ),
                StageInputBinding(
                    "flow_face_to_cells_path",
                    ".npy",
                    "flow_run/face_to_cells.npy",
                ),
                StageInputBinding(
                    "flow_cell_volumes_path",
                    ".npy",
                    "flow_run/cell_volumes.npy",
                ),
            ),
        ),
        StageId.REACTIVE_TRANSPORT.value: StageDefinition(
            StageId.REACTIVE_TRANSPORT,
            StageReactiveTransportParameters,
            (
                StageInputBinding(
                    "mesh_npz_path",
                    ".npz",
                    "mesh.npz",
                ),
                StageInputBinding(
                    "flow_summary_path",
                    ".json",
                    "flow_run/summary.json",
                ),
                StageInputBinding(
                    "flow_coupling_metadata_path",
                    ".json",
                    "flow_run/flow_coupling_metadata.json",
                ),
                StageInputBinding(
                    "flow_face_flux_path",
                    ".npy",
                    "flow_run/final_corrected_face_flux.npy",
                ),
                StageInputBinding(
                    "flow_face_to_cells_path",
                    ".npy",
                    "flow_run/face_to_cells.npy",
                ),
                StageInputBinding(
                    "flow_cell_volumes_path",
                    ".npy",
                    "flow_run/cell_volumes.npy",
                ),
                StageInputBinding(
                    "reactive_case_path",
                    ".json",
                    "reactive_case.json",
                ),
            ),
        ),
    }
)

def parse_stage_parameters(
    experiment_id: str,
    raw: object,
    *,
    input_prefixes: InputPrefixes = "",
) -> StageParameters:
    definition = STAGE_REGISTRY[experiment_id]
    return definition.parameters_model.from_dict(raw, input_prefixes=input_prefixes)  # type: ignore[attr-defined]
