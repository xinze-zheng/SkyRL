"""Tool-call parser registry.

Provides :func:`get_tool_parser` to instantiate a parser by name.
Parsers are registered in :data:`TOOL_PARSERS`.

To add a new parser:

1. Create a subclass of :class:`~.base.ToolCallParser`.
2. Add it to :data:`TOOL_PARSERS` below.
"""

from .base import NoOpParser, ToolCallParser
from .hermes import HermesParser

TOOL_PARSERS = {
    "hermes": HermesParser,
    "none": NoOpParser,
}


def get_tool_parser(name: str) -> ToolCallParser:
    """Instantiate a tool-call parser by name.

    Args:
        name: Parser name (key in :data:`TOOL_PARSERS`).

    Returns:
        A :class:`ToolCallParser` instance.

    Raises:
        ValueError: If *name* is not a registered parser.
    """
    cls = TOOL_PARSERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tool_call_parser {name!r}. "
            f"Available: {sorted(TOOL_PARSERS)}"
        )
    return cls()
