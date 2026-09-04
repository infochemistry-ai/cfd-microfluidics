"""Unknown tool arguments are refused here, as the compute API refuses them.

`@mcp.tool()` turns each tool function's signature into a pydantic model, and
mcp 1.28.1 builds that model on
`mcp.server.fastmcp.utilities.func_metadata.ArgModelBase`, whose config is a
hard-coded `ConfigDict(arbitrary_types_allowed=True)` - so pydantic's default
`extra="ignore"` applies and an argument the signature does not name is dropped
before the tool body, and therefore before `backend.app.stage_registry`, ever
sees it. `Tool.from_function` then publishes `arg_model.model_json_schema()`
verbatim, which carries no `additionalProperties`, so the schema does not
mention the rule either. An agent that misspells an optional argument gets a
silent run using the default while believing it asked for something else, and
the compute API answers `invalid_parameters` for the identical body.

The SDK offers no supported way to change this: neither `FastMCP.tool()`,
`FastMCP.add_tool()`, `ToolManager.add_tool()`, `Tool.from_function()` nor
`func_metadata()` takes an "extra" or "additionalProperties" argument, and no
hook is exposed on the produced `Tool`. The lowlevel server *can* validate
arguments against the published schema - `Server.call_tool(validate_input=True)`
runs `jsonschema.validate` against `inputSchema` - but `FastMCP._setup_handlers`
deliberately registers its handler with `validate_input=False` so FastMCP can
keep pre-parsing JSON-encoded arguments first; re-enabling it would mean
reaching into `_mcp_server` and would reject the coerced values that
`FuncMetadata.pre_parse_json` exists to accept.

So the check lives here, on `FastMCP.call_tool` - the one public method every
tool call of every transport passes through, and the only place that still sees
the raw argument dict - while `list_tools` publishes the
`additionalProperties: false` that lets a well-behaved client catch the same
mistake a round trip earlier. Both halves are derived from the one published
schema, so the promise and the enforcement cannot drift apart.
"""

from __future__ import annotations

from typing import Any, Sequence

from mcp.server.fastmcp import FastMCP
from mcp.types import ContentBlock, Tool as MCPTool

from backend.app.stage_registry import StageParametersError

from .errors import tool_error


def unsupported_parameters_message(unknown: Sequence[str]) -> str:
    """The wording `backend/app/stage_registry.py` uses for the same refusal.

    Kept identical on purpose: the two doors of this process must answer the
    same body the same way, and `test_strict_arguments.py` compares this
    message against the one `parse_stage_parameters` actually raises.
    """

    return (
        "Unsupported parameter(s): " + ", ".join(repr(item) for item in unknown) + "."
    )


class StrictArgumentsFastMCP(FastMCP):
    """A FastMCP whose tools reject arguments their schema does not name."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._published_arguments: dict[str, frozenset[str]] = {}

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        for tool in tools:
            schema = tool.inputSchema
            if not isinstance(schema, dict):  # pragma: no cover - typed dict
                continue
            schema["additionalProperties"] = False
            properties = schema.get("properties")
            self._published_arguments[tool.name] = frozenset(
                properties if isinstance(properties, dict) else ()
            )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        allowed = await self._published_argument_names(name)
        # An unknown tool name is left to the tool manager, which owns that
        # error ("Unknown tool: <name>"); guessing it here would mean two
        # spellings of the same failure.
        if allowed is not None:
            unknown = sorted(set(arguments) - allowed)
            if unknown:
                raise tool_error(
                    StageParametersError(unsupported_parameters_message(unknown))
                )
        return await super().call_tool(name, arguments)

    async def _published_argument_names(self, name: str) -> frozenset[str] | None:
        """What the tool's own `inputSchema` says the caller may send.

        Tools are registered before the server serves anything, so this is a
        cache miss once; the refresh keeps a tool registered later from being
        enforced against a stale listing.
        """

        if name not in self._published_arguments:
            await self.list_tools()
        return self._published_arguments.get(name)
