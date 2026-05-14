"""Tests for TITO tool-call parsers."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.tool_parsers import get_tool_parser
from skyrl.backends.skyrl_train.inference_servers.tito.tool_parsers.base import NoOpParser, ToolCallParser
from skyrl.backends.skyrl_train.inference_servers.tito.tool_parsers.hermes import HermesParser


class TestToolParserRegistry:
    """Tests for the parser registry."""

    def test_get_hermes(self):
        p = get_tool_parser("hermes")
        assert isinstance(p, HermesParser)

    def test_get_none(self):
        p = get_tool_parser("none")
        assert isinstance(p, NoOpParser)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown tool_call_parser"):
            get_tool_parser("nonexistent")


class TestNoOpParser:
    """Tests for NoOpParser."""

    def test_returns_full_text(self):
        p = NoOpParser()
        result = p.parse("hello world")
        assert result["content"] == "hello world"
        assert result["tool_calls"] is None

    def test_empty_text(self):
        p = NoOpParser()
        result = p.parse("")
        assert result["content"] == ""
        assert result["tool_calls"] is None


class TestHermesParser:
    """Tests for HermesParser."""

    def test_no_tool_calls(self):
        p = HermesParser()
        result = p.parse("Just some text without tool calls")
        assert result["content"] == "Just some text without tool calls"
        assert result["tool_calls"] is None

    def test_single_tool_call(self):
        p = HermesParser()
        text = 'Some content\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
        result = p.parse(text)
        assert result["content"] == "Some content"
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "bash"
        assert '"command": "ls"' in tc["function"]["arguments"]
        assert tc["id"].startswith("call_")

    def test_tool_call_with_thinking(self):
        p = HermesParser()
        text = (
            "<think>Let me run a command</think>\n\n"
            '<tool_call>\n{"name": "bash", "arguments": {"command": "pwd"}}\n</tool_call>'
        )
        result = p.parse(text)
        assert "<think>" in result["content"]
        assert len(result["tool_calls"]) == 1

    def test_multiple_tool_calls(self):
        p = HermesParser()
        text = (
            '<tool_call>\n{"name": "f1", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "f2", "arguments": {"x": 1}}\n</tool_call>'
        )
        result = p.parse(text)
        assert result["content"] is None
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "f1"
        assert result["tool_calls"][1]["function"]["name"] == "f2"

    def test_malformed_json_returns_content(self):
        p = HermesParser()
        text = "<tool_call>\nnot valid json\n</tool_call>"
        result = p.parse(text)
        assert result["tool_calls"] is None
        # Content should be the text before <tool_call> or full text
        assert result["content"] is not None

    def test_partial_tool_call_at_eof(self):
        p = HermesParser()
        text = '<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls"}}'
        result = p.parse(text)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "bash"

    def test_content_before_tool_call(self):
        p = HermesParser()
        text = 'I will run the command.\n<tool_call>\n{"name": "exec", "arguments": {}}\n</tool_call>'
        result = p.parse(text)
        assert result["content"] == "I will run the command."

    def test_tool_call_only_no_content(self):
        p = HermesParser()
        text = '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
        result = p.parse(text)
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
