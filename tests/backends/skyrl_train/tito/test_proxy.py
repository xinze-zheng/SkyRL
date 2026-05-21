"""Integration tests for the TITO proxy handler."""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.config import TITOConfig
from skyrl.backends.skyrl_train.inference_servers.tito.proxy import TITOHandler, _build_app, _check_prefix
from skyrl.backends.skyrl_train.inference_servers.tito.session import SessionState
from skyrl.backends.skyrl_train.inference_servers.tito.tokenizer_backends import HttpTokenizerBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockTokenizerBackend:
    """Simple mock tokenizer backend for handler tests."""

    def __init__(self, prompt_tokens=None, delta_tokens=None, gen_prompt_tokens=None):
        self._prompt_tokens = prompt_tokens or [1, 2, 3, 4, 5]
        self._delta_tokens = delta_tokens or [10, 11, 12]
        self._gen_prompt_tokens = gen_prompt_tokens or [100, 101]

    async def tokenize_messages(self, model, messages, *, add_generation_prompt=True, tools=None):
        return list(self._prompt_tokens)

    async def tokenize_delta(self, model, new_messages, *, tools=None):
        return list(self._delta_tokens)

    async def get_gen_prompt_ids(self, model, messages, *, tools=None):
        return list(self._gen_prompt_tokens)


class MockHttpResponse:
    """Minimal mock for httpx response."""

    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


# ---------------------------------------------------------------------------
# _check_prefix tests
# ---------------------------------------------------------------------------


class TestCheckPrefix:
    """Tests for the prefix checking utility."""

    def test_empty_prefix(self):
        assert _check_prefix("s1", 0, [], [1, 2, 3]) is True

    def test_matching_prefix(self):
        assert _check_prefix("s1", 0, [1, 2, 3], [1, 2, 3, 4, 5]) is True

    def test_exact_match(self):
        assert _check_prefix("s1", 0, [1, 2], [1, 2]) is True

    def test_mismatch(self):
        assert _check_prefix("s1", 0, [1, 2, 3], [1, 2, 4, 5]) is False

    def test_cur_shorter_than_prev(self):
        assert _check_prefix("s1", 0, [1, 2, 3], [1, 2]) is False


# ---------------------------------------------------------------------------
# TITOHandler tests
# ---------------------------------------------------------------------------


class TestTITOHandler:
    """Tests for the TITOHandler request processing."""

    def _make_handler(self, completion_response=None, tokenizer=None):
        """Create a handler with mocked dependencies."""
        sessions: Dict[str, SessionState] = {}
        config = TITOConfig(prefix_check=False)
        mock_tokenizer = tokenizer or MockTokenizerBackend()

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Default completion response
        if completion_response is None:
            completion_response = {
                "choices": [{
                    "text": "Hello, I can help!",
                    "finish_reason": "stop",
                    "token_ids": [200, 201, 202],
                    "logprobs": None,
                }]
            }

        mock_client.post = AsyncMock(
            return_value=MockHttpResponse(completion_response)
        )

        handler = TITOHandler(
            backend_url="http://mock:8000",
            config=config,
            tokenizer_backend=mock_tokenizer,
            sessions=sessions,
            http_client=mock_client,
        )
        return handler, sessions

    @pytest.mark.asyncio
    async def test_first_turn(self):
        """First turn should tokenize full prompt and bookkeep."""
        handler, sessions = self._make_handler()
        req = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 100,
        }

        resp = await handler.handle_chat_completion("session1", req)
        assert resp.status_code == 200

        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"] == "Hello, I can help!"
        assert body["choices"][0]["finish_reason"] == "stop"

        # Session should be created with tokens
        assert "session1" in sessions
        s = sessions["session1"]
        assert s.turn == 1
        assert s.messages_seen == 2
        # prompt (5) + response (3)
        assert len(s.tokens) == 8
        assert s.loss_mask[:5] == [0, 0, 0, 0, 0]
        assert s.loss_mask[5:] == [1, 1, 1]
        assert len(s.transitions) == 1
        transition = s.transitions[0]
        assert transition.step == 0
        assert transition.input_token_ids.to_list() == [1, 2, 3, 4, 5, 100, 101]
        assert transition.output_token_ids == [200, 201, 202]
        assert transition.output_text == "Hello, I can help!"
        assert transition.assistant_message == {
            "role": "assistant",
            "reasoning_content": None,
            "content": "Hello, I can help!",
        }
        assert transition.observation_token_ids == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_second_turn(self):
        """Second turn should tokenize delta only."""
        handler, sessions = self._make_handler()

        # First turn
        req1 = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 100,
        }
        await handler.handle_chat_completion("s1", req1)

        # Second turn (agent added assistant + tool response)
        req2 = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Let me check"},
                {"role": "tool", "content": "result: OK"},
            ],
            "max_tokens": 100,
        }
        resp = await handler.handle_chat_completion("s1", req2)
        assert resp.status_code == 200

        s = sessions["s1"]
        assert s.turn == 2
        assert s.messages_seen == 4
        # prompt(5) + resp1(3) + delta(3) + resp2(3)
        assert len(s.tokens) == 14
        assert len(s.transitions) == 2
        assert s.transitions[1].step == 1
        assert s.transitions[1].input_token_ids.to_list() == [
            1, 2, 3, 4, 5, 200, 201, 202, 10, 11, 12, 100, 101,
        ]
        assert s.transitions[1].observation_token_ids == [10, 11, 12]

    @pytest.mark.asyncio
    async def test_tool_calls_parsed(self):
        """Tool calls in response should be parsed."""
        completion_response = {
            "choices": [{
                "text": '<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls"}}\n</tool_call>',
                "finish_reason": "stop",
                "token_ids": [300, 301, 302],
                "logprobs": None,
            }]
        }
        handler, sessions = self._make_handler(completion_response=completion_response)
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "run ls"}],
            "max_tokens": 50,
        }
        resp = await handler.handle_chat_completion("s1", req)
        body = json.loads(resp.body)
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        assert body["choices"][0]["message"]["tool_calls"] is not None
        assert len(body["choices"][0]["message"]["tool_calls"]) == 1
        assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"
        assert sessions["s1"].transitions[0].assistant_message["tool_calls"] is not None

    @pytest.mark.asyncio
    async def test_transition_records_logprobs(self):
        """Transition should capture completion logprobs when the backend returns them."""
        completion_response = {
            "choices": [{
                "text": "ok",
                "finish_reason": "stop",
                "token_ids": [300, 301],
                "logprobs": {
                    "tokens": ["o", "k"],
                    "token_logprobs": [-0.1, -0.2],
                    "top_logprobs": [{"o": -0.1}, {"k": -0.2}],
                },
            }]
        }
        handler, sessions = self._make_handler(completion_response=completion_response)
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50,
        }
        resp = await handler.handle_chat_completion("s1", req)
        assert resp.status_code == 200

        transition = sessions["s1"].transitions[0]
        assert transition.output_logprobs == [-0.1, -0.2]
        assert transition.output_top_logprobs == [{"o": -0.1}, {"k": -0.2}]

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Errors during processing should return 500."""
        sessions: Dict[str, SessionState] = {}
        config = TITOConfig(prefix_check=False)

        # Tokenizer that raises
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize_messages = AsyncMock(side_effect=RuntimeError("tokenize failed"))

        handler = TITOHandler(
            backend_url="http://mock:8000",
            config=config,
            tokenizer_backend=mock_tokenizer,
            sessions=sessions,
            http_client=AsyncMock(spec=httpx.AsyncClient),
        )

        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        resp = await handler.handle_chat_completion("s1", req)
        assert resp.status_code == 500
        body = json.loads(resp.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_max_completion_tokens_passthrough(self):
        """max_completion_tokens should be forwarded as max_tokens."""
        handler, sessions = self._make_handler()
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 256,
        }
        await handler.handle_chat_completion("s1", req)
        # Verify the post was called with max_tokens
        post_call = handler._client.post.call_args
        payload = post_call[1]["json"]
        assert payload["max_tokens"] == 256


class TestTITOTransitionEndpoints:
    """Tests for transition query endpoints."""

    def _make_app_with_session(self):
        app = _build_app("http://mock:8000", TITOConfig(prefix_check=False))
        session = SessionState(model="m")
        first = session.record_transition(
            step=0,
            input_token_ids=[1, 2, 3],
            output_token_ids=[4, 5],
            output_logprobs=[-0.1, -0.2],
            output_top_logprobs=[{"4": -0.1}, {"5": -0.2}],
            output_text="hello",
            assistant_message={"role": "assistant", "content": "hello"},
            observation_token_ids=[1, 2, 3],
        )
        second = session.record_transition(
            step=1,
            input_token_ids=[1, 2, 3, 4, 5, 6],
            output_token_ids=[7],
            output_logprobs=[-0.3],
            output_top_logprobs=[{"7": -0.3}],
            output_text="next",
            assistant_message={"role": "assistant", "content": "next"},
            observation_token_ids=[6],
        )
        app.state.sessions["s1"] = session
        return app, first, second

    @pytest.mark.asyncio
    async def test_list_transitions_compact(self):
        app, first, _ = self._make_app_with_session()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/session/s1/transitions")

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "s1"
        assert body["num_transitions"] == 2
        assert "input_token_prefix_tree" not in body
        assert body["transitions"][0]["input_prefix_hash"] == first.input_token_ids.prefix_hash
        assert "input_token_ids" not in body["transitions"][0]

    @pytest.mark.asyncio
    async def test_list_transitions_with_materialized_inputs_and_tree(self):
        app, _, second = self._make_app_with_session()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/session/s1/transitions?include_input_tokens=true&include_prefix_tree=true"
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "input_token_prefix_tree" in body
        assert body["transitions"][1]["input_token_ids"] == [1, 2, 3, 4, 5, 6]
        assert body["transitions"][1]["input_prefix_hash"] == second.input_token_ids.prefix_hash

    @pytest.mark.asyncio
    async def test_get_transition_by_step(self):
        app, _, _ = self._make_app_with_session()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/session/s1/transition/1?include_input_tokens=true")

        assert resp.status_code == 200
        transition = resp.json()["transition"]
        assert transition["step"] == 1
        assert transition["input_token_ids"] == [1, 2, 3, 4, 5, 6]
        assert transition["output_token_ids"] == [7]

    @pytest.mark.asyncio
    async def test_get_transition_by_prefix_hash(self):
        app, first, _ = self._make_app_with_session()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/session/s1/transition/by-prefix/{first.input_token_ids.prefix_hash}"
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["input_prefix_hash"] == first.input_token_ids.prefix_hash
        assert body["num_matches"] == 1
        assert body["transitions"][0]["step"] == 0

    @pytest.mark.asyncio
    async def test_get_prefix_by_hash(self):
        app, _, second = self._make_app_with_session()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/session/s1/prefix/{second.input_token_ids.prefix_hash}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["prefix_hash"] == second.input_token_ids.prefix_hash
        assert body["input_token_len"] == 6
        assert body["input_token_ids"] == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_transition_endpoint_missing_values(self):
        app, _, _ = self._make_app_with_session()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing_step = await client.get("/session/s1/transition/99")
            missing_hash = await client.get("/session/s1/transition/by-prefix/sha256:missing")
            missing_session = await client.get("/session/missing/transitions")

        assert missing_step.status_code == 404
        assert missing_hash.status_code == 404
        assert missing_session.status_code == 404
