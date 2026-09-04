"""Adapters that bridge local API calls to CFD stage scripts."""

from .base import AdapterRunOutcome, ComputeAdapter
from .local_stage_adapter import LocalStageAdapter

__all__ = [
    "AdapterRunOutcome",
    "ComputeAdapter",
    "LocalStageAdapter",
]
