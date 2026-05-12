"""Abstract base class for tool-call parsers.

Tool-call parsers extract structured tool calls from raw model output
text. Each parser implements :meth:`parse`, which splits the text into
``content`` (text shown to the user) and ``tool_calls`` (structured
function calls forwarded to the agent framework).

To add a new parser, subclass :class:`ToolCallParser`, implement
:meth:`parse`, and register it in ``tool_parsers/__init__.py``.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ToolCallParser(ABC):
    """Base class for extracting tool calls from raw model output.

    Subclasses must implement :meth:`parse`.
    """

    @abstractmethod
    def parse(self, text: str) -> Dict[str, Any]:
        """Extract tool calls from raw model output *text*.

        Args:
            text: Raw text from ``/v1/completions`` response, including
                any ``<think>`` blocks, content, and tool-call markup.

        Returns:
            Dict with two keys:

            - ``"content"`` (``str | None``): Text content before/outside
              tool calls. ``None`` if the entire output is tool calls.
            - ``"tool_calls"`` (``list[dict] | None``): Extracted tool
              calls in OpenAI format, or ``None`` if no tool calls found.
              Each dict has: ``{"id": str, "type": "function",
              "function": {"name": str, "arguments": str}}``.
        """


class NoOpParser(ToolCallParser):
    """Pass-through parser that never extracts tool calls.

    Returns the full text as ``content`` with ``tool_calls=None``.
    Useful when the agent framework handles tool-call parsing itself.
    """

    def parse(self, text: str) -> Dict[str, Any]:
        return {"content": text, "tool_calls": None}
