"""Strict reactive-case validation tool."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from microfluidics.reactive.case import (
    REACTIVE_CASE_CONTRACT_VERSION,
    reactive_case_from_mapping,
)

from .context import ServerContext
from .errors import guard

MAX_REACTIVE_CASE_BYTES = 1_048_576


def _validate_size(reactive_case: dict[str, Any]) -> None:
    try:
        size = len(
            json.dumps(
                reactive_case,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reactive_case must be JSON-compatible: {exc}") from exc
    if size > MAX_REACTIVE_CASE_BYTES:
        raise ValueError(
            "reactive_case_too_large: reactive_case exceeds the 1 MiB limit"
        )


def _validated(reactive_case: dict[str, Any]):
    _validate_size(reactive_case)
    return reactive_case_from_mapping(reactive_case, source_label="reactive_case")


def validate_reactive_case(
    ctx: ServerContext,
    *,
    reactive_case: dict[str, Any],
) -> dict[str, Any]:
    _ = ctx
    with guard():
        normalized = _validated(reactive_case)
    return {
        "valid": True,
        "contract_version": REACTIVE_CASE_CONTRACT_VERSION,
        "schema_version": normalized.schema_version,
        "case_id": normalized.case_id,
        "mode": normalized.mode,
        "sha256": normalized.reactive_case_sha256,
        "species": list(normalized.species_names),
        "reaction_count": len(normalized.mechanism.reactions),
    }


def register(mcp: FastMCP, ctx: ServerContext) -> None:
    @mcp.tool(
        name="cfd_validate_reactive_case",
        description=(
            "Strictly validate and normalize a reactive_case_v1 object without "
            "saving it. Returns its canonical SHA-256, species, and reaction count."
        ),
    )
    def cfd_validate_reactive_case(
        reactive_case: dict[str, Any],
    ) -> dict[str, Any]:
        return validate_reactive_case(ctx, reactive_case=reactive_case)

    _ = cfd_validate_reactive_case
