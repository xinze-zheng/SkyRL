"""Tests for TITO tokenizer backends."""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.tokenizer_backends import (
    DUMMY_BASE,
    HttpTokenizerBackend,
    RendererTokenizerBackend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockResponse:
    """Minimal httpx.Response mock."""

    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


def _make_tokenize_fn(token_map: Dict[str, List[int]]):
    """Create a mock tokenize function that maps message content to tokens.

    Args:
        token_map: Maps serialized message list → token IDs.
    """

    async def mock_post(url: str, json: dict = None, timeout: int = 30) -> MockResponse:
        messages = json.get("messages", []) if isinstance(json, dict) else []
        add_gen = json.get("add_generation_prompt", False) if isinstance(json, dict) else False
        key = str([(m.get("role"), m.get("content", "")[:20]) for m in messages]) + f"_gen={add_gen}"
        tokens = token_map.get(key, list(range(len(messages) * 10)))
        return MockResponse({"tokens": tokens})

    return mock_post


# ---------------------------------------------------------------------------
# HttpTokenizerBackend tests
# ---------------------------------------------------------------------------


class TestHttpTokenizerBackend:
    """Tests for the HTTP-based tokenizer backend."""

    @pytest.mark.asyncio
    async def test_tokenize_messages(self):
        """Should call /tokenize and return tokens."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=MockResponse({"tokens": [1, 2, 3]}))

        backend = HttpTokenizerBackend("http://backend:8000", client)
        result = await backend.tokenize_messages("model", [{"role": "user", "content": "hi"}])
        assert result == [1, 2, 3]
        client.post.assert_called_once()
        call_args = client.post.call_args
        assert "/tokenize" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_tokenize_messages_with_tools(self):
        """Should forward tools in the payload."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=MockResponse({"tokens": [1]}))

        backend = HttpTokenizerBackend("http://backend:8000", client)
        tools = [{"type": "function", "function": {"name": "test"}}]
        await backend.tokenize_messages("model", [], tools=tools)
        payload = client.post.call_args[1]["json"]
        assert payload["tools"] == tools

    @pytest.mark.asyncio
    async def test_tokenize_delta(self):
        """Delta tokenization should subtract dummy base tokens."""
        call_count = 0
        responses = [
            MockResponse({"tokens": [10, 20, 30]}),  # DUMMY_BASE
            MockResponse({"tokens": [10, 20, 30, 40, 50]}),  # DUMMY_BASE + new
        ]

        client = AsyncMock(spec=httpx.AsyncClient)

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        client.post = AsyncMock(side_effect=side_effect)

        backend = HttpTokenizerBackend("http://backend:8000", client)
        result = await backend.tokenize_delta("model", [{"role": "tool", "content": "output"}])
        assert result == [40, 50]

    @pytest.mark.asyncio
    async def test_get_gen_prompt_ids(self):
        """Gen prompt IDs should be the diff of with/without gen prompt."""
        call_count = 0
        responses = [
            MockResponse({"tokens": [1, 2, 3, 4, 5]}),  # with gen prompt
            MockResponse({"tokens": [1, 2, 3]}),  # without gen prompt
        ]

        client = AsyncMock(spec=httpx.AsyncClient)

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        client.post = AsyncMock(side_effect=side_effect)

        backend = HttpTokenizerBackend("http://backend:8000", client)
        result = await backend.get_gen_prompt_ids("model", [{"role": "user", "content": "hi"}])
        assert result == [4, 5]


# ---------------------------------------------------------------------------
# RendererTokenizerBackend tests
# ---------------------------------------------------------------------------


class TestRendererTokenizerBackend:
    """Tests for the renderer-based tokenizer backend."""

    def _make_mock_renderer(self):
        """Create a mock renderer with predictable token outputs."""
        renderer = MagicMock()
        renderer.render_ids = MagicMock(return_value=[1, 2, 3])
        renderer.parse_response = MagicMock(
            return_value=MagicMock(content="hello", reasoning_content=None, tool_calls=None)
        )
        renderer.bridge_to_next_turn = MagicMock(return_value=MagicMock(token_ids=[1, 2, 3, 4, 5]))
        return renderer

    @pytest.mark.asyncio
    async def test_tokenize_messages(self):
        renderer = self._make_mock_renderer()
        renderer.render_ids.return_value = [10, 20, 30]
        http_fallback = MagicMock(spec=HttpTokenizerBackend)
        backend = RendererTokenizerBackend(renderer, http_fallback)

        result = await backend.tokenize_messages(
            "model", [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )
        assert result == [10, 20, 30]
        renderer.render_ids.assert_called_once()

    @pytest.mark.asyncio
    async def test_tokenize_delta(self):
        renderer = self._make_mock_renderer()
        dummy_tokens = [1, 2, 3]
        full_tokens = [1, 2, 3, 4, 5, 6]
        renderer.render_ids.side_effect = [dummy_tokens, full_tokens]

        http_fallback = MagicMock(spec=HttpTokenizerBackend)
        backend = RendererTokenizerBackend(renderer, http_fallback)

        result = await backend.tokenize_delta("model", [{"role": "tool", "content": "out"}])
        assert result == [4, 5, 6]

    @pytest.mark.asyncio
    async def test_get_gen_prompt_ids(self):
        renderer = self._make_mock_renderer()
        renderer.render_ids.side_effect = [
            [1, 2, 3, 100, 101],  # with gen prompt
            [1, 2, 3],  # without gen prompt
        ]
        http_fallback = MagicMock(spec=HttpTokenizerBackend)
        backend = RendererTokenizerBackend(renderer, http_fallback)

        result = await backend.get_gen_prompt_ids("model", [{"role": "user", "content": "hi"}])
        assert result == [100, 101]

    def test_bridge_to_next_turn(self):
        renderer = self._make_mock_renderer()
        renderer.bridge_to_next_turn.return_value = MagicMock(token_ids=[1, 2, 3, 4, 5, 6, 7])
        http_fallback = MagicMock(spec=HttpTokenizerBackend)
        backend = RendererTokenizerBackend(renderer, http_fallback)

        result = backend.bridge_to_next_turn(
            [1, 2, 3], [4, 5], [{"role": "tool", "content": "result"}]
        )
        assert result == [1, 2, 3, 4, 5, 6, 7]

    def test_bridge_to_next_turn_returns_none(self):
        renderer = self._make_mock_renderer()
        renderer.bridge_to_next_turn.return_value = None
        http_fallback = MagicMock(spec=HttpTokenizerBackend)
        backend = RendererTokenizerBackend(renderer, http_fallback)

        result = backend.bridge_to_next_turn([1, 2], [3, 4], [{"role": "tool", "content": "x"}])
        assert result is None

    def test_parse_response_no_tool_calls(self):
        renderer = self._make_mock_renderer()
        renderer.parse_response.return_value = MagicMock(
            content="just text", reasoning_content=None, tool_calls=None
        )
        http_fallback = MagicMock(spec=HttpTokenizerBackend)
        backend = RendererTokenizerBackend(renderer, http_fallback)

        result = backend.parse_response([10, 20, 30])
        assert result["content"] == "just text"
        assert result["tool_calls"] is None

    def test_parse_response_with_reasoning(self):
        renderer = self._make_mock_renderer()
        renderer.parse_response.return_value = MagicMock(
            content="answer", reasoning_content="thinking...", tool_calls=None
        )
        http_fallback = MagicMock(spec=HttpTokenizerBackend)
        backend = RendererTokenizerBackend(renderer, http_fallback)

        result = backend.parse_response([10, 20])
        assert "<think>thinking...</think>" in result["content"]
        assert "answer" in result["content"]
