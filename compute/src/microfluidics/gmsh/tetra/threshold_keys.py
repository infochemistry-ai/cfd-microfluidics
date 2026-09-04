"""Threshold-key helpers with canonical + legacy scientific labels."""

from __future__ import annotations

from typing import Any, Mapping

OUTLET_FRACTION_THRESHOLDS: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2)


def canonical_scientific_label(value: float) -> str:
    """Return stable scientific labels, e.g. 1e-4 instead of 1e-04."""
    label = f"{float(value):.0e}"
    mantissa, exponent = label.split("e", maxsplit=1)
    return f"{mantissa}e{int(exponent)}"


def legacy_scientific_label(value: float) -> str:
    """Return historical scientific labels with zero-padded exponent."""
    return f"{float(value):.0e}"


def format_threshold_key(
    prefix: str, threshold: float, *, canonical: bool = True
) -> str:
    label = (
        canonical_scientific_label(threshold)
        if canonical
        else legacy_scientific_label(threshold)
    )
    return f"{prefix}_{label}"


def threshold_key_aliases(prefix: str, threshold: float) -> tuple[str, ...]:
    canonical_key = format_threshold_key(prefix, threshold, canonical=True)
    legacy_key = format_threshold_key(prefix, threshold, canonical=False)
    if canonical_key == legacy_key:
        return (canonical_key,)
    return (canonical_key, legacy_key)


def read_threshold_value(
    metrics: Mapping[str, Any],
    prefix: str,
    threshold: float,
    default: float = 0.0,
) -> float:
    for key in threshold_key_aliases(prefix, threshold):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return float(default)
