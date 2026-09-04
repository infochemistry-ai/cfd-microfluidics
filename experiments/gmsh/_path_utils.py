"""Compatibility shim for experiment path helpers."""

from __future__ import annotations

from pathlib import Path

from microfluidics.path_contract import (
    create_timestamped_run_dir as _create_timestamped_run_dir,
    normalize_user_path as _normalize_user_path_impl,
)


def _normalize_user_path(value: str | Path) -> Path:
    return _normalize_user_path_impl(value)


def normalize_user_path(value: str | Path) -> Path:
    return _normalize_user_path_impl(value)


def create_timestamped_run_dir(base_dir: Path, stem: str) -> Path:
    return _create_timestamped_run_dir(base_dir, stem)


def option_was_explicitly_provided(raw_argv: list[str], option: str) -> bool:
    """Return whether an argparse option appeared as ``--x v`` or ``--x=v``."""
    return any(item == option or item.startswith(f"{option}=") for item in raw_argv)
