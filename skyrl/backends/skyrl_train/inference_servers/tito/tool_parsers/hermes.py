"""Hermes-format tool-call parser.

Extracts tool calls delimited by ``<tool_call>`` / ``</tool_call>`` XML
tags, as used by Hermes-2-Pro, Qwen3, and related models.

Example model output::

    <think>reasoning...</think>

    Some content text
    <tool_call>
    {"name": "bash", "arguments": {"command": "ls -la"}}
    </tool_call>

Parsed result::

    {
        "content": "<think>reasoning...</think>\\n\\nSome content text",
        "tool_calls": [{
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\\"command\\": \\"ls -la\\"}"}
        }]
    }
"""

import json
import re
import uuid
from typing import Any, Dict, List

from .base import ToolCallParser

_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
)


class HermesParser(ToolCallParser):
    """Parse ``<tool_call>`` XML tags from raw model output.

    Handles both complete (``<tool_call>...</tool_call>``) and
    partial (``<tool_call>...EOF``) tool calls. Multiple tool calls
    in a single response are supported.

    Content before the first ``<tool_call>`` tag is returned as
    ``content``. If JSON parsing fails for any tool call, the entire
    output is returned as plain content with ``tool_calls=None``.
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """Extract hermes-format tool calls from *text*.

        Args:
            text: Raw model output that may contain ``<tool_call>`` tags.

        Returns:
            Dict with ``content`` and ``tool_calls`` keys.
            See :meth:`ToolCallParser.parse` for format details.
        """
        if "<tool_call>" not in text:
            return {"content": text, "tool_calls": None}

        content = text[: text.index("<tool_call>")].strip() or None
        matches = _TOOL_CALL_RE.findall(text)

        tool_calls: List[Dict[str, Any]] = []
        for complete, partial in matches:
            raw = complete if complete else partial
            try:
                parsed = json.loads(raw.strip())
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": parsed.get("name", ""),
                            "arguments": json.dumps(
                                parsed.get("arguments", {}),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            except (json.JSONDecodeError, KeyError):
                # Malformed tool call — treat entire output as content
                if content is None:
                    content = text
                return {"content": content, "tool_calls": None}

        return {"content": content, "tool_calls": tool_calls or None}
