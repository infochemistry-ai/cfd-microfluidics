"""Run the five CFD stages as isolated local subprocesses."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from threading import Event
from typing import Any

from microfluidics_contracts import RuntimeSettings, SubmitRunRequestV1
from microfluidics.path_contract import normalize_user_path, resolve_service_runs_root

from backend.app.input_errors import StageInputError
from backend.app.stage_registry import (
    STAGE_REGISTRY,
    Device,
    StageDefinition,
    StageParameters,
    parse_stage_parameters,
    stage_input_prefixes,
)

from .base import AdapterRunOutcome
from .local_process import build_subprocess_env, run_local_process
from .stage_commands import (
    StageLayout,
    build_stage_solver_argv,
    local_stage_layout,
)

# Inside the per-run work directory: staged inputs and solver results.
WORK_SUBDIR = "work"
OUTPUTS_SUBDIR = "outputs"

# The one stage input that is itself a list of further inputs.
FLOW_SUMMARY_PARAMETER = "flow_summary_path"


def require_compute_device(device: Device) -> None:
    """Fail before staging when an explicitly requested CUDA device is unusable."""

    if device is not Device.CUDA:
        return
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CUDA was requested, but PyTorch is not installed. Install a "
            "CUDA-enabled PyTorch 2.5.1 build first."
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA was requested, but no usable CUDA device is visible. Check "
            "the NVIDIA driver, the CUDA-enabled PyTorch build, and "
            "CUDA_VISIBLE_DEVICES."
        )


def resolve_stage_input_path(
    project_root: Path,
    value: str,
    *,
    parameter_name: str,
) -> Path:
    """Resolve one stage input to an existing file inside the project root.

    `validate_input_path` has already refused absolute paths, "..", backslashes and
    control characters by the time a request reaches an adapter. This is the
    check that key cannot make: it resolves symlinks, so a link inside the
    repository pointing outside it is rejected here rather than followed.
    """

    raw = str(value).strip()
    if not raw:
        raise StageInputError(
            f"'{parameter_name}' must be a non-empty repository-relative path.",
            code="input_not_found",
            http_status=404,
        )
    normalized = normalize_user_path(raw).expanduser()
    if normalized.is_absolute():
        resolved = normalized.resolve()
    else:
        resolved = (project_root / normalized).resolve()
    if not resolved.is_relative_to(project_root):
        raise StageInputError(
            f"'{parameter_name}' must resolve inside the project root.",
            code="input_outside_project_root",
            http_status=400,
        )
    if not resolved.is_file():
        raise StageInputError(
            f"'{parameter_name}' does not name a file inside the project root.",
            code="input_not_found",
            http_status=404,
        )
    return resolved


def _reference_basename(value: str) -> str:
    """The last component of a path written under either separator convention.

    A flow summary records absolute paths from the machine that produced it,
    and that machine may be a Windows one: `PurePosixPath` sees no separator in
    `C:\\runs\\flow_1\\summary.json` and hands the whole string back as its
    `.name`, so the absolute reference survives the reduction and thermal reads
    the original flow run instead of its staged inputs. `PureWindowsPath`
    accepts both `\\` and `/`, so one pass over it covers summaries from either
    platform, whichever platform is doing the staging.

    A POSIX file name may legitimately contain a backslash, and such a name is
    over-reduced here. That direction is safe: the result is still a bare name
    resolved beside the staged summary, so the worst case is a missing file
    rather than a silent read of the wrong one.
    """

    return PureWindowsPath(value).name


def localize_flow_summary_references(staged_summary: Path) -> None:
    """Point a staged flow summary's artifact references at its own directory.

    `stage_thermal` is the only stage whose entrypoint is handed a file that
    names *more* files: `experiments/gmsh/_flow_coupling.py` reads
    `artifacts.flow_coupling_metadata_json` out of the summary and makes that
    file's directory the flow run directory for all five coupling arrays. A
    flow run records those artifacts as absolute paths on the machine that
    produced them, and `_resolve_reference_path` prefers the absolute value
    whenever it still exists, falling back to the same basename beside the
    summary only when it does not.

    In an isolated staged run the original run directory is never present, so the
    fallback always wins and thermal reads exactly what was staged. On a
    checkout it usually *is* present - and the summary is a caller-supplied
    file - so the absolute reference would win instead: thermal would read a
    different flow run than the one whose artifacts the caller named, or, with
    a hand-written summary, files anywhere on this disk. That is the one input
    path `resolve_stage_input_path`'s containment check cannot reach, because
    it checks the summary, not the paths inside it.

    Reducing every reference to its basename removes the choice: resolution
    can only land in the staged flow run directory.

    The reduction reads both separator conventions - see `_reference_basename`.
    """

    try:
        payload = json.loads(staged_summary.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageInputError(
            f"'{FLOW_SUMMARY_PARAMETER}' must be a readable JSON flow summary.",
            code="invalid_flow_summary",
            http_status=400,
        ) from exc
    if not isinstance(payload, dict):
        raise StageInputError(
            f"'{FLOW_SUMMARY_PARAMETER}' must contain a JSON object.",
            code="invalid_flow_summary",
            http_status=400,
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    payload["artifacts"] = {
        key: (_reference_basename(value) if isinstance(value, str) else value)
        for key, value in artifacts.items()
    }
    try:
        staged_summary.write_text(
            json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
        )
    except OSError as exc:
        raise StageInputError(
            f"Unable to stage CFD input {FLOW_SUMMARY_PARAMETER!r}.",
            code="input_staging_unavailable",
            http_status=507,
        ) from exc


def stage_local_inputs(
    *,
    definition: StageDefinition,
    parameters: StageParameters,
    project_root: Path,
    layout: StageLayout,
    cancel_event: Event,
) -> dict[str, str]:
    """Copy every declared input to the staged name the solver expects.

    The solver reads a private staged copy under names it can rely on and can never
    write over the caller's file. Returns parameter name -> staged path.
    """

    staged: dict[str, str] = {}
    for binding in definition.inputs:
        if cancel_event.is_set():
            raise InterruptedError("CFD stage preparation was cancelled.")
        source = resolve_stage_input_path(
            project_root,
            str(getattr(parameters, binding.parameter_name)),
            parameter_name=binding.parameter_name,
        )
        destination = Path(layout.staged(binding.staged_path))
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        except OSError as exc:
            raise StageInputError(
                f"Unable to stage CFD input {binding.parameter_name!r}.",
                code="input_staging_unavailable",
                http_status=507,
            ) from exc
        if binding.parameter_name == FLOW_SUMMARY_PARAMETER:
            localize_flow_summary_references(destination)
        staged[binding.parameter_name] = destination.as_posix()
    return staged


class LocalStageAdapter:
    """Executes registered CFD stages as local subprocesses."""

    name = "local-stage-adapter"

    def __init__(
        self,
        project_root: Path,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.settings = settings or RuntimeSettings.from_env()
        self._require_chainable_run_root()

    def _require_chainable_run_root(self) -> None:
        """Refuse a run root whose outputs could not feed the next stage.

        Artifacts are handed back as references the agent copies verbatim into
        stage N+1, and `validate_input_path` rejects any key beginning with "/".
        A SERVICE_RUN_ROOT that resolves outside the project root can only
        produce absolute references, so stage N would succeed and stage N+1
        would refuse its own predecessor's output with 'invalid_parameters' -
        the chain this adapter exists to serve is broken before it starts.

        There is nothing an individual stage could do about it, so the
        configuration is refused where the adapter is built (service startup)
        instead of one stage later. An absolute SERVICE_RUN_ROOT is still fine
        as long as it lands inside the project root.
        """

        run_root = resolve_service_runs_root(
            self.project_root,
            self.settings.service_run_root,
        )
        if run_root.is_relative_to(self.project_root):
            return
        raise ValueError(
            "SERVICE_RUN_ROOT must resolve inside the project root: local "
            "stage artifacts are handed back as "
            "repository-relative paths and the next stage cannot accept an "
            f"absolute one. {self.settings.service_run_root!r} resolves to "
            f"{run_root}, outside {self.project_root}."
        )

    def _resolve_definition(self, experiment_id: str) -> StageDefinition:
        definition = STAGE_REGISTRY.get(experiment_id)
        if definition is None:
            allowed = ", ".join(sorted(STAGE_REGISTRY))
            raise ValueError(
                f"Unknown experiment_id={experiment_id!r}. Allowed values: {allowed}."
            )
        return definition

    def _reference(self, path: Path) -> str:
        """A reference an agent can pass back: a repository-relative POSIX path.

        `_require_chainable_run_root` has already refused a SERVICE_RUN_ROOT
        outside the project root, and stage entrypoints and staged inputs live
        under it too, so in practice every path reaching here is inside the
        checkout. The absolute fallback stays for the metadata fields this also
        formats (`script`, `stage_work_root`), where an unexpected path is
        better reported verbatim than turned into a wrong relative one; it
        cannot appear among the artifacts a next stage is offered.
        """

        resolved = path.resolve()
        if resolved.is_relative_to(self.project_root):
            return resolved.relative_to(self.project_root).as_posix()
        return resolved.as_posix()

    def _collect_artifacts(self, output_root: Path) -> list[str]:
        """Every file the stage left under its output root.

        Only the output root is walked: the staged input copies are the
        caller's own files under another name, and reporting them would offer
        a second candidate for basenames like mesh.npz and make the next
        stage's parameter ambiguous.
        """

        if not output_root.is_dir():
            return []
        return sorted(
            self._reference(path) for path in output_root.rglob("*") if path.is_file()
        )

    def run(
        self,
        run_id: str,
        request: SubmitRunRequestV1,
        cancel_event: Event,
        run_work_dir: Path,
    ) -> AdapterRunOutcome:
        _ = run_id
        definition = self._resolve_definition(request.experiment_id)
        parameters = parse_stage_parameters(
            request.experiment_id,
            request.parameters,
            input_prefixes=stage_input_prefixes(self.settings),
        )
        device = getattr(parameters, "device", Device.CPU)
        require_compute_device(device)
        normalized_parameters = parameters.to_dict()

        started = datetime.now(timezone.utc)
        run_work_dir.mkdir(parents=True, exist_ok=True)
        work_root = run_work_dir / WORK_SUBDIR
        output_root = run_work_dir / OUTPUTS_SUBDIR
        output_root.mkdir(parents=True, exist_ok=True)
        layout = local_stage_layout(
            project_root=self.project_root,
            work_root=work_root,
            output_root=output_root,
        )

        staged = stage_local_inputs(
            definition=definition,
            parameters=parameters,
            project_root=self.project_root,
            layout=layout,
            cancel_event=cancel_event,
        )
        command = build_stage_solver_argv(request.experiment_id, parameters, layout)
        script = Path(command[1])
        if not script.exists():
            raise FileNotFoundError(f"Stage entrypoint not found: {script}")

        timeout_seconds = float(self.settings.run_timeout_seconds)
        log_path = run_work_dir / "compute.log"
        # MPLBACKEND matches build_job_payload: a service subprocess has no
        # display, and stage scripts render preview figures.
        env = build_subprocess_env(self.project_root, MPLBACKEND="Agg")

        with log_path.open("w", encoding="utf-8") as log_file:
            process_outcome = run_local_process(
                command,
                cwd=self.project_root,
                env=env,
                log_file=log_file,
                cancel_event=cancel_event,
                timeout_seconds=timeout_seconds,
            )

        finished = datetime.now(timezone.utc)
        metadata: dict[str, Any] = {
            "script": self._reference(script),
            "stage_work_root": self._reference(work_root),
            "staged_inputs": {
                name: self._reference(Path(path)) for name, path in staged.items()
            },
            "cancelled": process_outcome.cancelled,
            "timed_out": process_outcome.timed_out,
            "timeout_seconds": timeout_seconds if process_outcome.timed_out else None,
            "request_parameters": normalized_parameters,
            "output_root": self._reference(output_root),
        }

        return AdapterRunOutcome(
            exit_code=process_outcome.exit_code,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_seconds=max(0.0, (finished - started).total_seconds()),
            artifacts=self._collect_artifacts(output_root),
            metadata=metadata,
            log_path=self._reference(log_path),
        )
