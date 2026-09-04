"""Standalone local compute execution service."""

from __future__ import annotations

import logging
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable

from microfluidics.path_contract import (
    create_timestamped_run_dir,
    resolve_service_runs_root,
)
from microfluidics_contracts import (
    ErrorPayloadV1,
    ExecutionResponseV1,
    ResultPayloadV1,
    RunStatus,
    RuntimeSettings,
    SubmitRunRequestV1,
)

from .input_errors import StageInputError
from .request_validation import prepare_submit_request
from .worker.runner import ComputeRunner


logger = logging.getLogger(__name__)

_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

_MAX_IDEMPOTENCY_RECORDS = 4096


class RunCapacityError(Exception):
    """Raised when no background execution slot is free for a new submission."""


class IdempotencyConflictError(Exception):
    """The request_id was already claimed by a different normalized request."""


@dataclass
class _IdempotencyRecord:
    fingerprint: str
    ready: Event
    run_id: str = ""
    response: ExecutionResponseV1 | None = None
    error: Exception | None = None


def _safe_request_stem(request_id: str) -> str:
    safe = _UNSAFE_STEM_CHARS.sub("-", request_id.strip()).strip("._-")
    return (safe or "request")[:80]


class ComputeExecutionService:
    """Executes one compute request and returns a terminal response inline."""

    def __init__(
        self,
        *,
        project_root: Path,
        settings: RuntimeSettings | None = None,
        runner: ComputeRunner | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.settings = settings or RuntimeSettings.from_env()
        self.runner = runner or ComputeRunner(
            project_root=self.project_root,
            settings=self.settings,
        )
        self.service_runs_root = resolve_service_runs_root(
            self.project_root,
            self.settings.service_run_root,
        )
        self._idempotency_lock = Lock()
        self._idempotency: dict[str, _IdempotencyRecord] = {}
        self._active_cancel_events: dict[str, Event] = {}
        self._max_concurrent_runs = max(
            1, int(self.settings.service_max_concurrent_runs)
        )
        self._active_run_count = 0

    def _prune_idempotency_locked(self, *, target_size: int) -> None:
        """Discard oldest completed responses while preserving active requests."""

        if len(self._idempotency) <= target_size:
            return
        for request_id, record in list(self._idempotency.items()):
            if len(self._idempotency) <= target_size:
                break
            if record.ready.is_set():
                self._idempotency.pop(request_id, None)

    @staticmethod
    def _request_fingerprint(request: SubmitRunRequestV1) -> str:
        normalized = json.dumps(
            {
                "experiment_id": request.experiment_id,
                "parameters": request.parameters,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def execute(self, request: SubmitRunRequestV1) -> ExecutionResponseV1:
        prepared, record, owner = self._claim(request)
        if record is not None and not owner:
            return self._await_claimed(record)
        return self._execute_claimed(prepared, record)

    def _claim(
        self,
        request: SubmitRunRequestV1,
    ) -> tuple[SubmitRunRequestV1, "_IdempotencyRecord | None", bool]:
        """Validate, then take ownership of the request_id or find its owner."""

        request = prepare_submit_request(self.project_root, self.settings, request)
        if request.request_id is None:
            return request, None, True
        fingerprint = self._request_fingerprint(request)
        with self._idempotency_lock:
            existing = self._idempotency.get(request.request_id)
            if existing is None:
                self._prune_idempotency_locked(target_size=_MAX_IDEMPOTENCY_RECORDS - 1)
                record = _IdempotencyRecord(fingerprint=fingerprint, ready=Event())
                self._idempotency[request.request_id] = record
                return request, record, True
            if existing.fingerprint != fingerprint:
                raise IdempotencyConflictError(
                    "request_id is already associated with a different "
                    "normalized request."
                )
            if existing.ready.is_set() and self._is_failed_outcome(existing):
                record = _IdempotencyRecord(fingerprint=fingerprint, ready=Event())
                self._idempotency[request.request_id] = record
                return request, record, True
            return request, existing, False

    @staticmethod
    def _is_failed_outcome(record: "_IdempotencyRecord") -> bool:
        """Whether a ready record holds a failure, i.e. is safe to re-run.

        Only SUCCEEDED and CANCELLED outcomes are cached: success is the
        idempotency guarantee, and a cancellation was explicitly asked for and
        must not silently restart. Everything else is retryable."""

        if record.error is not None:
            return True
        if record.response is None:
            # Unreachable by construction: `_release_record` is the only thing
            # that sets `ready`, and it substitutes a terminal FAILED response
            # first, so a ready record always carries one. Kept because the
            # alternative to this branch is an AttributeError on the next line
            # if that invariant is ever broken, and because a record that
            # somehow has no response cannot be replayed at all - _await_claimed
            # would assert and get() would hand back None - so re-running is
            # the only safe answer.
            return True
        return record.response.status is RunStatus.FAILED

    def _await_claimed(self, record: "_IdempotencyRecord") -> ExecutionResponseV1:
        record.ready.wait()
        with self._idempotency_lock:
            error = record.error
            response = record.response
        if error is not None:
            raise error
        assert response is not None
        return response

    def _execute_claimed(
        self,
        request: SubmitRunRequestV1,
        record: "_IdempotencyRecord | None",
        cancel_event: Event | None = None,
    ) -> ExecutionResponseV1:
        """Run one claimed request, always leaving its record terminal.

        `cancel_event` is supplied when the caller registered one before this
        run became observable - see `submit_async`. Otherwise one is made here,
        which is the synchronous path: `execute()` has not returned to anybody
        yet, so nothing can cancel a run that is not running.
        """

        run_id = uuid.uuid4().hex
        if record is not None:
            with self._idempotency_lock:
                record.run_id = run_id
        request_id = request.request_id or run_id
        try:
            return self._run_to_terminal(
                request=request,
                record=record,
                run_id=run_id,
                request_id=request_id,
                cancel_event=cancel_event,
            )
        finally:
            self._release_record(
                record=record,
                request_id=request_id,
                run_id=run_id,
            )

    def _release_record(
        self,
        *,
        record: "_IdempotencyRecord | None",
        request_id: str,
        run_id: str,
        fallback_error: ErrorPayloadV1 | None = None,
    ) -> None:
        """Retire a claimed run's live state and make its record terminal.

        This is the only place `record.ready` is ever set, and every path that
        claims a record reaches it from a `finally`, so no exit can leave a
        record un-ready: not a `BaseException` (KeyboardInterrupt, SystemExit,
        a BaseExceptionGroup) escaping the runner, and not a background worker
        that could not be started at all. That state is permanent once
        reached - `get()` reports PENDING for ever, `_await_claimed` waits with
        no timeout, `_claim` cannot re-claim it and `_prune_idempotency_locked`
        refuses to evict it - so the request_id would be poisoned for the life
        of the process.

        A record arriving here without a response of its own is given a
        terminal FAILED one, which `_is_failed_outcome` treats as retryable.
        """

        with self._idempotency_lock:
            self._active_cancel_events.pop(request_id, None)
            if record is None or record.ready.is_set():
                return
            if record.response is None:
                record.response = ExecutionResponseV1(
                    request_id=request_id,
                    run_id=run_id,
                    status=RunStatus.FAILED,
                    result=None,
                    error=fallback_error
                    or ErrorPayloadV1(
                        code="compute_abandoned",
                        message=(
                            "Compute execution ended without recording an outcome."
                        ),
                    ),
                )
            record.ready.set()
            self._prune_idempotency_locked(target_size=_MAX_IDEMPOTENCY_RECORDS)

    def _run_to_terminal(
        self,
        *,
        request: SubmitRunRequestV1,
        record: "_IdempotencyRecord | None",
        run_id: str,
        request_id: str,
        cancel_event: Event | None = None,
    ) -> ExecutionResponseV1:
        cancel_event = cancel_event if cancel_event is not None else Event()

        try:
            run_work_dir = create_timestamped_run_dir(
                self.service_runs_root,
                f"{_safe_request_stem(request_id)}_{run_id[:8]}",
            )
            if request.request_id is not None:
                with self._idempotency_lock:
                    # A no-op when `submit_async` registered this very event
                    # before returning; the synchronous path registers here.
                    self._active_cancel_events[request_id] = cancel_event
            if cancel_event.is_set():
                raise InterruptedError("Compute run was cancelled before it started.")
            result = self.runner.run(
                run_id=run_id,
                request=request,
                cancel_event=cancel_event,
                run_work_dir=run_work_dir,
            )
            result.request_id = request_id
            status, error = self._terminal_outcome(
                result=result,
                cancel_event=cancel_event,
            )
        except InterruptedError:
            result = None
            status = RunStatus.CANCELLED
            error = None
        except StageInputError as exc:
            self._record_raised_failure(
                record=record,
                request_id=request_id,
                run_id=run_id,
                code=exc.code,
                exc=exc,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Synchronous compute execution failed run_id=%s", run_id)
            result = None
            status = RunStatus.FAILED
            error = ErrorPayloadV1(
                code="compute_exception",
                message=str(exc).strip() or exc.__class__.__name__,
                details={"error_type": exc.__class__.__name__},
            )

        response = ExecutionResponseV1(
            request_id=request_id,
            run_id=run_id,
            status=status,
            result=result,
            error=error,
        )
        if record is not None:
            with self._idempotency_lock:
                record.response = response
        return response

    def _record_raised_failure(
        self,
        *,
        record: "_IdempotencyRecord | None",
        request_id: str,
        run_id: str,
        code: str,
        exc: Exception,
    ) -> None:
        """Record a re-raised failure as an observable, retryable outcome.

        The exception still propagates so the synchronous HTTP path can map it
        onto its status code, but the record keeps a terminal FAILED response
        so an async caller polling get() sees the failure instead of a missing
        request. record.error stays set so a concurrent synchronous waiter in
        _await_claimed still gets the exception rather than a FAILED body.

        Only the outcome is recorded here; `_release_record` publishes it as
        the exception unwinds, so readiness has exactly one owner."""

        response = ExecutionResponseV1(
            request_id=request_id,
            run_id=run_id,
            status=RunStatus.FAILED,
            result=None,
            error=ErrorPayloadV1(
                code=code,
                message=str(exc).strip() or exc.__class__.__name__,
                details={"error_type": exc.__class__.__name__},
            ),
        )
        with self._idempotency_lock:
            if record is not None:
                record.error = exc
                record.response = response

    def submit_async(self, request: SubmitRunRequestV1) -> str:
        """Validate now, execute locally in the background, return request_id."""

        caller_supplied_request_id = request.request_id is not None
        if not caller_supplied_request_id:
            request = replace(request, request_id=uuid.uuid4().hex)
        prepared, record, owner = self._claim(request)
        request_id = str(prepared.request_id)
        if record is None or not owner:
            return request_id

        cancel_event = Event()
        with self._idempotency_lock:
            self._active_cancel_events[request_id] = cancel_event

        def _run() -> None:
            try:
                self._execute_claimed(prepared, record, cancel_event=cancel_event)
            except Exception:
                # Terminal state is already recorded by _execute_claimed; this
                # only prevents an unhandled exception from killing the thread.
                logger.exception(
                    "Background compute execution failed request_id=%s", request_id
                )
            finally:
                with self._idempotency_lock:
                    self._active_run_count -= 1

        try:
            self._start_background_run(record=record, request_id=request_id, work=_run)
        except RunCapacityError:
            if not caller_supplied_request_id:
                # This request_id was minted here, never handed to the
                # caller (RunCapacityError propagates instead of the
                # request_id), and cannot be polled - so leaving the
                # terminal record behind would only strand it until LRU
                # eviction. Drop it, unless something else has since claimed
                # the same id (a fresh uuid4, so effectively never).
                with self._idempotency_lock:
                    if self._idempotency.get(request_id) is record:
                        self._idempotency.pop(request_id, None)
            raise
        return request_id

    def _start_background_run(
        self,
        *,
        record: "_IdempotencyRecord",
        request_id: str,
        work: "Callable[[], None]",
    ) -> None:
        """Admit one background run against a fixed bound, or refuse it.

        An agent calls cfd_run_* as often as it likes and every accepted call
        spawns a solver - a subprocess of this process in standalone mode - so
        submissions must not be able to create workers without limit. At most
        SERVICE_MAX_CONCURRENT_RUNS of them are alive at a time; the bound is
        on live workers rather than on a queue, deliberately. A submission
        that arrives while the bound is reached is refused immediately with
        RunCapacityError and recorded as a terminal FAILED response the agent
        can also see through get(), because a request silently parked behind a
        long solver run is indistinguishable from one that started.
        Refused submissions keep the caller's request_id retryable.
        """

        error = ErrorPayloadV1(
            code="capacity_exceeded",
            message=(
                "The service is already executing its configured maximum of "
                f"{self._max_concurrent_runs} concurrent background runs "
                "(SERVICE_MAX_CONCURRENT_RUNS). Nothing was started; retry "
                "this request_id once a run finishes."
            ),
        )
        with self._idempotency_lock:
            admitted = self._active_run_count < self._max_concurrent_runs
            if admitted:
                self._active_run_count += 1
        if not admitted:
            # No run ever started, so there is no run_id from a runner - but
            # every other terminal record carries a real one (get() exposes
            # it, and an empty string reads as "still unassigned" rather than
            # "refused"), so one is minted here purely for the record.
            self._release_record(
                record=record,
                request_id=request_id,
                run_id=uuid.uuid4().hex,
                fallback_error=error,
            )
            raise RunCapacityError(error.message)
        try:
            Thread(
                target=work,
                name=f"compute-submit-{request_id[:16]}",
                daemon=True,
            ).start()
        except BaseException as exc:
            # Thread.start() raises RuntimeError when the OS refuses another
            # thread. The slot has to come back and the record has to become
            # terminal here: nothing else will ever run for it.
            with self._idempotency_lock:
                self._active_run_count -= 1
            self._release_record(
                record=record,
                request_id=request_id,
                run_id=uuid.uuid4().hex,
                fallback_error=ErrorPayloadV1(
                    code="capacity_exceeded",
                    message=(
                        "A worker for this run could not be started: "
                        f"{str(exc).strip() or exc.__class__.__name__}."
                    ),
                    details={"error_type": exc.__class__.__name__},
                ),
            )
            raise RunCapacityError(
                "A worker for this run could not be started."
            ) from exc

    def get(self, request_id: str) -> ExecutionResponseV1 | None:
        with self._idempotency_lock:
            record = self._idempotency.get(request_id)
            if record is None:
                return None
            if not record.ready.is_set():
                return ExecutionResponseV1(
                    request_id=request_id,
                    run_id=record.run_id,
                    status=RunStatus.PENDING,
                )
            return record.response

    def cancel(self, request_id: str) -> dict:
        """Signal the local subprocess for a known asynchronous request."""

        with self._idempotency_lock:
            record = self._idempotency.get(request_id)
            active_event = self._active_cancel_events.get(request_id)
            if active_event is not None:
                active_event.set()
        cancellation_requested = active_event is not None
        cancelled = [request_id] if cancellation_requested else []
        already_terminal = (
            [request_id]
            if record is not None and record.ready.is_set() and not cancellation_requested
            else []
        )
        return {
            "request_id": request_id,
            "cancelled": cancelled,
            "already_terminal": already_terminal,
            **({"cancellation_requested": True} if cancellation_requested else {}),
        }

    def _terminal_outcome(
        self,
        *,
        result: ResultPayloadV1,
        cancel_event: Event,
    ) -> tuple[RunStatus, ErrorPayloadV1 | None]:
        metadata = result.metadata or {}
        if (
            cancel_event.is_set()
            or metadata.get("cancelled")
            or result.exit_code == 130
        ):
            return RunStatus.CANCELLED, None
        if metadata.get("timed_out"):
            timeout_seconds = metadata.get(
                "timeout_seconds",
                self.settings.run_timeout_seconds,
            )
            return (
                RunStatus.FAILED,
                ErrorPayloadV1(
                    code="run_timeout",
                    message="Compute run exceeded configured timeout.",
                    details={
                        "timeout_seconds": timeout_seconds,
                        "exit_code": result.exit_code,
                    },
                ),
            )
        if result.exit_code != 0:
            return (
                RunStatus.FAILED,
                ErrorPayloadV1(
                    code="compute_nonzero_exit",
                    message="Compute process finished with non-zero exit code.",
                    details={"exit_code": result.exit_code},
                ),
            )
        return RunStatus.SUCCEEDED, None
