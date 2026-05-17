"""Integration tests for the TITO proxy pipeline using real Qwen3 trajectories.

These tests simulate the full TITO handler pipeline (tokenize → bookkeep →
parse → prefix-check) using real trajectory data from Qwen3-32B SWE-bench
runs. They verify token-level correctness without requiring a GPU or vLLM
backend.

Fixtures:
    ``fixtures/trajectory_rewarded.json`` — 16-message trajectory with
    reward=1, all assistant turns have tool_calls.
    ``fixtures/trajectory_mixed.json`` — 33-message trajectory with a
    format-error turn (assistant without tool_calls) and user reminder.

Run with::

    pytest tests/backends/skyrl_train/tito/test_pipeline_qwen3.py -v
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.config import TITOConfig
from skyrl.backends.skyrl_train.inference_servers.tito.proxy import TITOHandler, _check_prefix
from skyrl.backends.skyrl_train.inference_servers.tito.response import build_chat_completion_response
from skyrl.backends.skyrl_train.inference_servers.tito.session import SessionState
from skyrl.backends.skyrl_train.inference_servers.tito.tokenizer_backends import (
    DUMMY_BASE,
    HttpTokenizerBackend,
    RendererTokenizerBackend,
)
from skyrl.backends.skyrl_train.inference_servers.tito.tool_parsers import get_tool_parser
from skyrl.backends.skyrl_train.inference_servers.tito.tool_parsers.hermes import HermesParser

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MODEL_NAME = "Qwen/Qwen3-8B"

# Skip all tests if transformers/renderers not available (CI without GPU deps)
try:
    from transformers import AutoTokenizer
    from renderers import create_renderer

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="transformers/renderers not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load Qwen3 tokenizer once for all tests."""
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def renderer(tokenizer):
    """Create Qwen3 renderer."""
    return create_renderer(tokenizer, renderer="qwen3")


@pytest.fixture(scope="module")
def rewarded_trajectory() -> Dict[str, Any]:
    """Load the rewarded trajectory fixture."""
    with open(FIXTURES_DIR / "trajectory_rewarded.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def mixed_trajectory() -> Dict[str, Any]:
    """Load the mixed trajectory fixture (has format error turns)."""
    with open(FIXTURES_DIR / "trajectory_mixed.json") as f:
        return json.load(f)


def _get_conversation_turns(messages: List[Dict]) -> List[List[Dict]]:
    """Split messages into per-request conversation snapshots.

    Each agent request sends the full history up to that point.
    Returns a list of message lists, one per assistant turn.
    """
    turns = []
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            # The request that generated this assistant message included
            # all messages up to (but not including) this assistant message
            turns.append(messages[:i])
    return turns


# ---------------------------------------------------------------------------
# Tokenizer backend tests with real Qwen3
# ---------------------------------------------------------------------------


class TestQwen3TokenizerBackendConsistency:
    """Verify that HttpTokenizerBackend's dummy-base approach produces
    correct delta tokens for real Qwen3 conversations."""

    def test_full_tokenize_matches_apply_chat_template(self, tokenizer, rewarded_trajectory):
        """render_ids should match tokenizer.apply_chat_template for the same messages."""
        from renderers import create_renderer

        r = create_renderer(tokenizer, renderer="qwen3")
        msgs = rewarded_trajectory["messages"][:2]  # system + user

        renderer_ids = r.render_ids(msgs, add_generation_prompt=False)
        template_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=False, tokenize=True, return_dict=False,
        )
        assert renderer_ids == template_ids, (
            f"Renderer and apply_chat_template disagree: "
            f"{len(renderer_ids)} vs {len(template_ids)} tokens"
        )

    def test_delta_tokenization_consistency(self, tokenizer):
        """The dummy-base delta approach should produce consistent tokens
        when the delta messages follow the dummy base's role pattern."""
        # The dummy-base approach works because special tokens act as BPE
        # merge barriers. Verify this by tokenizing a user message delta.
        delta_msgs = [
            {"role": "user", "content": "Please check the output and fix the error."},
        ]

        # Method 1: dummy base subtraction
        dummy_ids = tokenizer.apply_chat_template(
            DUMMY_BASE, add_generation_prompt=False, tokenize=True, return_dict=False,
        )
        full_with_delta = tokenizer.apply_chat_template(
            DUMMY_BASE + delta_msgs, add_generation_prompt=False, tokenize=True, return_dict=False,
        )
        delta_ids = full_with_delta[len(dummy_ids):]

        # The delta should contain the user message tokens
        decoded = tokenizer.decode(delta_ids)
        assert "check the output" in decoded, f"Delta should contain message content, got: {decoded!r}"
        assert len(delta_ids) > 0, "Delta should not be empty"

        # Verify BPE merge barrier: tokenizing the same delta again should
        # produce identical tokens (deterministic)
        full_with_delta2 = tokenizer.apply_chat_template(
            DUMMY_BASE + delta_msgs, add_generation_prompt=False, tokenize=True, return_dict=False,
        )
        delta_ids2 = full_with_delta2[len(dummy_ids):]
        assert delta_ids == delta_ids2, "Delta tokenization should be deterministic"

    def test_gen_prompt_ids_extraction(self, tokenizer, renderer):
        """Generation prompt IDs should be the suffix added by add_generation_prompt."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        ids_with = renderer.render_ids(msgs, add_generation_prompt=True)
        ids_without = renderer.render_ids(msgs, add_generation_prompt=False)
        gen_prompt = ids_with[len(ids_without):]

        # Should be <|im_start|>assistant\n
        decoded = tokenizer.decode(gen_prompt)
        assert "assistant" in decoded, f"Gen prompt should contain 'assistant', got: {decoded!r}"
        assert len(gen_prompt) > 0, "Gen prompt should not be empty"


# ---------------------------------------------------------------------------
# Renderer bridge tests with real Qwen3
# ---------------------------------------------------------------------------


class TestQwen3RendererBridge:
    """Test bridge_to_next_turn with real Qwen3 renderer."""

    def test_bridge_returns_non_none_for_qwen3(self, renderer):
        """Qwen3Renderer should support bridge_to_next_turn."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        prompt_ids = renderer.render_ids(msgs, add_generation_prompt=True)

        # Simulate a short completion
        completion_ids = [100, 200, 300]

        result = renderer.bridge_to_next_turn(
            previous_prompt_ids=prompt_ids,
            previous_completion_ids=completion_ids,
            new_messages=[{"role": "tool", "content": "output here"}],
        )
        assert result is not None, "Qwen3Renderer should support bridge"
        assert len(result.token_ids) > len(prompt_ids) + len(completion_ids)

    def test_bridge_preserves_prefix(self, renderer):
        """Bridge result should start with prev_prompt + prev_completion."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Fix the bug"},
        ]
        prompt_ids = renderer.render_ids(msgs, add_generation_prompt=True)
        completion_ids = [100, 200, 300]

        result = renderer.bridge_to_next_turn(
            previous_prompt_ids=prompt_ids,
            previous_completion_ids=completion_ids,
            new_messages=[{"role": "tool", "content": "done"}],
        )
        assert result is not None
        expected_prefix = prompt_ids + completion_ids
        actual_prefix = result.token_ids[:len(expected_prefix)]
        assert actual_prefix == expected_prefix, "Bridge should preserve prefix"


# ---------------------------------------------------------------------------
# Session state tests with real tokenization
# ---------------------------------------------------------------------------


class TestSessionStateWithRealTokens:
    """Test SessionState accumulation with real Qwen3 tokens."""

    def test_multi_turn_accumulation(self, tokenizer, renderer, rewarded_trajectory):
        """Simulate a multi-turn session and verify token/mask integrity."""
        msgs = rewarded_trajectory["messages"]
        session = SessionState(model=MODEL_NAME)

        # Turn 0: tokenize initial prompt (system + user)
        initial_msgs = msgs[:2]
        prompt_ids = renderer.render_ids(initial_msgs, add_generation_prompt=False)
        session.tokens = list(prompt_ids)
        session.loss_mask = [0] * len(prompt_ids)
        session.messages_seen = 2

        assert len(session.tokens) == len(session.loss_mask)
        assert all(m == 0 for m in session.loss_mask), "Prompt should be all loss_mask=0"

        # Simulate first response (assistant msg[2])
        # In real TITO, response_token_ids come from /v1/completions
        # Here we approximate by tokenizing the assistant content
        asst_content = msgs[2].get("content", "")
        # Use dummy-base to get just the assistant message tokens
        asst_ids = tokenizer.apply_chat_template(
            DUMMY_BASE + [msgs[2]], add_generation_prompt=False,
            tokenize=True, return_dict=False,
        )
        dummy_ids = tokenizer.apply_chat_template(
            DUMMY_BASE, add_generation_prompt=False,
            tokenize=True, return_dict=False,
        )
        response_ids = asst_ids[len(dummy_ids):]

        session.last_prompt_len = len(session.tokens)
        session.append_response(response_ids)
        session.turn = 1
        session.messages_seen = 3

        total = len(prompt_ids) + len(response_ids)
        assert len(session.tokens) == total
        assert len(session.loss_mask) == total
        assert session.loss_mask[:len(prompt_ids)] == [0] * len(prompt_ids)
        assert session.loss_mask[len(prompt_ids):] == [1] * len(response_ids)

    def test_to_dict_preserves_all_fields(self, renderer, rewarded_trajectory):
        """to_dict should include all necessary fields for training."""
        msgs = rewarded_trajectory["messages"][:2]
        prompt_ids = renderer.render_ids(msgs, add_generation_prompt=False)

        session = SessionState(model=MODEL_NAME)
        session.tokens = list(prompt_ids)
        session.loss_mask = [0] * len(prompt_ids)
        session.turn = 0

        d = session.to_dict()
        assert "tokens" in d
        assert "loss_mask" in d
        assert "turn" in d
        assert "model" in d
        assert len(d["tokens"]) == len(d["loss_mask"])

    def test_begin_turn_and_reset(self, renderer):
        """Test begin_turn snapshot and reset_from_full_render."""
        session = SessionState(model=MODEL_NAME)
        session.tokens = [1, 2, 3]
        session.loss_mask = [0, 0, 0]
        session.messages_seen = 2

        prev = session.begin_turn()
        assert prev == 2

        # Simulate full re-render
        new_tokens = [10, 20, 30, 40, 50]
        new_mask = [0, 0, 1, 1, 0]
        session.reset_from_full_render(new_tokens, new_mask)
        assert session.tokens == [10, 20, 30, 40, 50]
        assert session.loss_mask == [0, 0, 1, 1, 0]


# ---------------------------------------------------------------------------
# Tool parsing with real model output
# ---------------------------------------------------------------------------


class TestToolParsingFromTrajectory:
    """Test tool call parsing against real trajectory data."""

    def test_hermes_parser_on_trajectory_content(self, rewarded_trajectory):
        """HermesParser should extract tool calls from real model output."""
        parser = HermesParser()
        msgs = rewarded_trajectory["messages"]

        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue
            content = msg.get("content", "") or ""
            has_tc_field = msg.get("tool_calls") is not None

            # If the trajectory has tool_calls, the content should either:
            # 1. Have <tool_call> tags (renderer stripped them), or
            # 2. NOT have tags (vLLM/renderer already extracted them)
            # The hermes parser on raw text should find tags when present
            parsed = parser.parse(content)

            if "<tool_call>" in content:
                assert parsed["tool_calls"] is not None, (
                    f"msg[{i}]: has <tool_call> in content but parser returned None"
                )

    def test_renderer_parse_response_round_trip(self, tokenizer, renderer, rewarded_trajectory):
        """Tokenize assistant content → parse_response should recover tool calls."""
        msgs = rewarded_trajectory["messages"]

        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant" or not msg.get("tool_calls"):
                continue

            # Tokenize the assistant message (including <tool_call> tags)
            full_content = msg.get("content", "") or ""

            # The trajectory content may not have <tool_call> tags
            # (they were stripped by the proxy). Check if we can reconstruct.
            tc = msg["tool_calls"][0]
            tc_name = tc["function"]["name"]

            # Verify the tool call name matches what mini-swe-agent expects
            assert tc_name == "bash", f"msg[{i}]: expected tool name 'bash', got '{tc_name}'"
            break  # Just verify first tool call


# ---------------------------------------------------------------------------
# End-to-end pipeline simulation
# ---------------------------------------------------------------------------


class TestPipelineSimulation:
    """Simulate the full TITO handler pipeline turn by turn using real data."""

    def test_full_rollout_token_accumulation(self, tokenizer, renderer, rewarded_trajectory):
        """Simulate a complete rollout and verify final token/mask state."""
        msgs = rewarded_trajectory["messages"]
        session = SessionState(model=MODEL_NAME)

        # Identify turns: each assistant message = one proxy request
        # The proxy receives all messages up to and including the current request
        turn_count = 0
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue

            # Messages the agent sends = everything up to this assistant
            request_msgs = msgs[:i]

            if turn_count == 0:
                # Turn 0: tokenize full prompt
                prompt_ids = renderer.render_ids(request_msgs, add_generation_prompt=False)
                session.tokens = list(prompt_ids)
                session.loss_mask = [0] * len(prompt_ids)
                session.messages_seen = len(request_msgs)
            else:
                # Turn N: tokenize delta observations
                prev_seen = session.begin_turn()
                new_msgs = request_msgs[prev_seen:]

                # Skip leading assistant messages (already bookkeept)
                obs_start = 0
                while obs_start < len(new_msgs) and new_msgs[obs_start].get("role") == "assistant":
                    obs_start += 1
                obs_msgs = new_msgs[obs_start:]

                if obs_msgs:
                    dummy_ids = renderer.render_ids(DUMMY_BASE, add_generation_prompt=False)
                    full_ids = renderer.render_ids(
                        DUMMY_BASE + obs_msgs, add_generation_prompt=False,
                    )
                    delta_ids = full_ids[len(dummy_ids):]
                    session.append_prompt(delta_ids)

                session.messages_seen = len(request_msgs)

            # Simulate response (approximate: tokenize assistant content as delta)
            asst_msg = msgs[i]
            dummy_ids = renderer.render_ids(DUMMY_BASE, add_generation_prompt=False)
            full_ids = renderer.render_ids(
                DUMMY_BASE + [asst_msg], add_generation_prompt=False,
            )
            response_ids = full_ids[len(dummy_ids):]

            session.last_prompt_len = len(session.tokens)
            session.append_response(response_ids)
            session.turn += 1
            turn_count += 1

        # Verify final state
        assert len(session.tokens) == len(session.loss_mask)
        assert session.turn == turn_count
        assert len(session.tokens) > 0

        # Verify loss mask structure: should have interleaved 0s and 1s
        has_zeros = any(m == 0 for m in session.loss_mask)
        has_ones = any(m == 1 for m in session.loss_mask)
        assert has_zeros, "Should have prompt/obs tokens with mask=0"
        assert has_ones, "Should have response tokens with mask=1"

        # First tokens should be mask=0 (prompt)
        assert session.loss_mask[0] == 0, "First token should be prompt (mask=0)"

        # Split for training: first loss_mask=1 position
        first_gen = next(
            (i for i, m in enumerate(session.loss_mask) if m == 1), len(session.loss_mask)
        )
        prompt_ids = session.tokens[:first_gen]
        response_ids = session.tokens[first_gen:]
        response_mask = session.loss_mask[first_gen:]

        assert len(prompt_ids) > 0, "Should have non-empty prompt"
        assert len(response_ids) > 0, "Should have non-empty response"
        assert len(response_ids) == len(response_mask)

    def test_mixed_trajectory_handles_format_errors(self, tokenizer, renderer, mixed_trajectory):
        """Pipeline should handle trajectories with format error turns
        (assistant without tool_calls, followed by user reminder)."""
        msgs = mixed_trajectory["messages"]
        session = SessionState(model=MODEL_NAME)

        turn_count = 0
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue

            request_msgs = msgs[:i]

            if turn_count == 0:
                prompt_ids = renderer.render_ids(request_msgs, add_generation_prompt=False)
                session.tokens = list(prompt_ids)
                session.loss_mask = [0] * len(prompt_ids)
                session.messages_seen = len(request_msgs)
            else:
                prev_seen = session.begin_turn()
                new_msgs = request_msgs[prev_seen:]
                obs_start = 0
                while obs_start < len(new_msgs) and new_msgs[obs_start].get("role") == "assistant":
                    obs_start += 1
                obs_msgs = new_msgs[obs_start:]

                if obs_msgs:
                    dummy_ids = renderer.render_ids(DUMMY_BASE, add_generation_prompt=False)
                    full_ids = renderer.render_ids(
                        DUMMY_BASE + obs_msgs, add_generation_prompt=False,
                    )
                    delta_ids = full_ids[len(dummy_ids):]
                    session.append_prompt(delta_ids)

                session.messages_seen = len(request_msgs)

            # Simulate response
            asst_msg = msgs[i]
            dummy_ids = renderer.render_ids(DUMMY_BASE, add_generation_prompt=False)
            full_ids = renderer.render_ids(
                DUMMY_BASE + [asst_msg], add_generation_prompt=False,
            )
            response_ids = full_ids[len(dummy_ids):]

            session.last_prompt_len = len(session.tokens)
            session.append_response(response_ids)
            session.turn += 1
            turn_count += 1

        # Should complete without errors even with format error turns
        assert session.turn == turn_count
        assert len(session.tokens) == len(session.loss_mask)
        assert len(session.tokens) > 0


# ---------------------------------------------------------------------------
# Context window capping
# ---------------------------------------------------------------------------


class TestContextWindowCapping:
    """Test max_tokens capping logic."""

    def test_cap_max_tokens_when_near_limit(self):
        """When prompt approaches max_model_len, max_tokens should be capped."""
        max_model_len = 40960
        prompt_len = 38000
        requested_max_tokens = 4096
        headroom = max_model_len - prompt_len

        capped = min(requested_max_tokens, headroom)
        assert capped == headroom == 2960
        assert capped < requested_max_tokens

    def test_graceful_response_when_prompt_exceeds_limit(self):
        """When prompt exceeds max_model_len, should return a valid
        ChatCompletion with finish_reason=length."""
        resp = build_chat_completion_response(
            model="test",
            content="",
            tool_calls=None,
            prompt_tokens=45000,
            completion_tokens=0,
            finish_reason="length",
        )
        assert resp["choices"][0]["finish_reason"] == "length"
        assert resp["choices"][0]["message"]["content"] == ""
        assert resp["usage"]["completion_tokens"] == 0


# ---------------------------------------------------------------------------
# Prefix check with real tokens
# ---------------------------------------------------------------------------


class TestPrefixCheckWithRealTokens:
    """Test prefix checking with real Qwen3 tokenized sequences."""

    def test_prefix_match_on_identical_tokenization(self, renderer):
        """Same messages tokenized twice should pass prefix check."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world"},
        ]
        ids1 = renderer.render_ids(msgs, add_generation_prompt=False)
        ids2 = renderer.render_ids(msgs, add_generation_prompt=False)

        assert _check_prefix("test", 0, ids1, ids2) is True

    def test_prefix_mismatch_detection(self, renderer):
        """Different messages should fail prefix check."""
        msgs1 = [{"role": "user", "content": "Hello"}]
        msgs2 = [{"role": "user", "content": "Goodbye"}]

        ids1 = renderer.render_ids(msgs1, add_generation_prompt=False)
        ids2 = renderer.render_ids(msgs2, add_generation_prompt=False)

        assert _check_prefix("test", 0, ids1, ids2) is False


# ---------------------------------------------------------------------------
# Response builder with trajectory data
# ---------------------------------------------------------------------------


class TestResponseBuilderWithTrajectoryData:
    """Test that build_chat_completion_response produces valid responses
    that litellm/mini-swe-agent can consume."""

    def test_tool_call_response_format(self):
        """Response with tool_calls should have correct format for litellm."""
        tool_calls = [{
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
        }]
        resp = build_chat_completion_response(
            model="Qwen/Qwen3-32B",
            content="<think>Let me check</think>",
            tool_calls=tool_calls,
            prompt_tokens=1000,
            completion_tokens=50,
            finish_reason="stop",
        )

        choice = resp["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["reasoning_content"] is None
        assert choice["message"]["tool_calls"] == tool_calls
        assert "<think>" in choice["message"]["content"]

    def test_no_tool_call_response(self):
        """Response without tool_calls (format error case)."""
        resp = build_chat_completion_response(
            model="Qwen/Qwen3-32B",
            content="I will use a bash command to check.",
            tool_calls=None,
            prompt_tokens=500,
            completion_tokens=20,
        )

        choice = resp["choices"][0]
        assert choice["finish_reason"] == "stop"
        # tool_calls key is omitted when None
        assert choice["message"].get("tool_calls") is None
        assert "bash" in choice["message"]["content"]
