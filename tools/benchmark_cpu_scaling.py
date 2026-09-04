"""Run a stage command with controlled CPU thread counts.

The command is executed in a fresh process for every sample so OpenMP and BLAS
read the requested limits before NumPy or PyTorch is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


THREAD_ENV_NAMES = (
    "MICROFLUIDICS_CPU_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("thread counts must be positive integers")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end CPU scaling for one stage command.",
    )
    parser.add_argument(
        "--stage", choices=("flow", "transport", "thermal"), required=True
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=_positive_int,
        default=(1, 2, 4, 8),
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Stage command after --, for example: -- uv run ...",
    )
    return parser


def _thread_environment(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for name in THREAD_ENV_NAMES:
        env[name] = str(threads)
    env["MICROFLUIDICS_TORCH_INTEROP_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def _run_once(
    command: list[str],
    *,
    cwd: Path,
    threads: int,
    timeout_seconds: float | None,
) -> float:
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=_thread_environment(threads),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    elapsed = perf_counter() - started
    if completed.returncode != 0:
        stderr_tail = completed.stderr[-4000:].strip()
        raise RuntimeError(
            f"benchmark command failed for {threads} threads with exit code "
            f"{completed.returncode}:\n{stderr_tail}"
        )
    return elapsed


def main() -> int:
    args = _parser().parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a stage command must be provided after --")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")

    cwd = args.working_directory.resolve()
    results: list[dict[str, object]] = []
    for threads in dict.fromkeys(args.threads):
        print(f"[cpu-scaling] stage={args.stage} threads={threads} warmup")
        for _ in range(args.warmups):
            _run_once(
                command,
                cwd=cwd,
                threads=threads,
                timeout_seconds=args.timeout_seconds,
            )

        samples: list[float] = []
        for repeat in range(1, args.repeats + 1):
            elapsed = _run_once(
                command,
                cwd=cwd,
                threads=threads,
                timeout_seconds=args.timeout_seconds,
            )
            samples.append(elapsed)
            print(
                f"[cpu-scaling] stage={args.stage} threads={threads} "
                f"repeat={repeat}/{args.repeats} elapsed={elapsed:.6f}s"
            )
        results.append(
            {
                "threads": threads,
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
                "min_seconds": min(samples),
                "max_seconds": max(samples),
            }
        )

    one_thread = next(
        (float(item["median_seconds"]) for item in results if item["threads"] == 1),
        None,
    )
    for item in results:
        median = float(item["median_seconds"])
        item["speedup_vs_one_thread"] = (
            one_thread / median if one_thread is not None else None
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "command": command,
        "working_directory": str(cwd),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "results": results,
        "measurement_scope": "end_to_end_subprocess_wall_time",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[cpu-scaling] wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
