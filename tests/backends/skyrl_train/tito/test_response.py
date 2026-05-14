"""Tests for TITO chat completion response builder."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.response import build_chat_completion_response


class TestBuildChatCompletionResponse:
    """Tests for build_chat_completion_response."""

    def test_basic_response(self):
        resp = build_chat_completion_response(
            model="test-model",
            content="Hello world",
            tool_calls=None,
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert resp["object"] == "chat.completion"
        assert resp["model"] == "test-model"
        assert len(resp["choices"]) == 1
        msg = resp["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello world"
        assert msg["reasoning_content"] is None
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["usage"]["prompt_tokens"] == 10
        assert resp["usage"]["completion_tokens"] == 5
        assert resp["usage"]["total_tokens"] == 15

    def test_tool_calls_override_finish_reason(self):
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
            }
        ]
        resp = build_chat_completion_response(
            model="m",
            content=None,
            tool_calls=tool_calls,
            prompt_tokens=5,
            completion_tokens=3,
            finish_reason="stop",
        )
        assert resp["choices"][0]["finish_reason"] == "tool_calls"
        assert resp["choices"][0]["message"]["tool_calls"] == tool_calls
        assert resp["choices"][0]["message"]["content"] is None

    def test_none_content(self):
        resp = build_chat_completion_response(
            model="m",
            content=None,
            tool_calls=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert resp["choices"][0]["message"]["content"] is None

    def test_logprobs_conversion(self):
        logprobs_data = {
            "tokens": ["Hello", " world"],
            "token_logprobs": [-0.5, -1.0],
            "top_logprobs": [{"Hello": -0.5, "Hi": -1.0}, {" world": -1.0}],
        }
        resp = build_chat_completion_response(
            model="m",
            content="Hello world",
            tool_calls=None,
            prompt_tokens=1,
            completion_tokens=2,
            logprobs_data=logprobs_data,
        )
        lp = resp["choices"][0]["logprobs"]
        assert lp is not None
        assert len(lp["content"]) == 2
        assert lp["content"][0]["token"] == "Hello"
        assert lp["content"][0]["logprob"] == -0.5
        assert len(lp["content"][0]["top_logprobs"]) == 2

    def test_no_logprobs(self):
        resp = build_chat_completion_response(
            model="m",
            content="x",
            tool_calls=None,
            prompt_tokens=1,
            completion_tokens=1,
        )
        assert resp["choices"][0]["logprobs"] is None

    def test_response_has_id(self):
        resp = build_chat_completion_response(
            model="m",
            content="x",
            tool_calls=None,
            prompt_tokens=1,
            completion_tokens=1,
        )
        assert resp["id"].startswith("chatcmpl-")

    def test_length_finish_reason(self):
        resp = build_chat_completion_response(
            model="m",
            content="truncated",
            tool_calls=None,
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="length",
        )
        assert resp["choices"][0]["finish_reason"] == "length"

    def test_reasoning_content_none_prevents_litellm_parsing(self):
        """reasoning_content=None should be explicitly set to prevent
        litellm from auto-parsing <think> blocks."""
        resp = build_chat_completion_response(
            model="m",
            content="<think>reasoning</think>\n\nresult",
            tool_calls=None,
            prompt_tokens=1,
            completion_tokens=1,
        )
        msg = resp["choices"][0]["message"]
        assert msg["reasoning_content"] is None
        assert "<think>" in msg["content"]
