"""Integration tests for the TITO proxy pipeline with GLM-4.7 (GLM-4.5 renderer).

Validates the TITO pipeline using the GLM-4.5 renderer against a synthetic
trajectory. Covers tokenization consistency, bridge_to_next_turn, session
accumulation, and tool parsing — all on CPU.

Run with::

    pytest tests/backends/skyrl_train/tito/test_pipeline_glm47.py -v
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.proxy import _check_prefix
from skyrl.backends.skyrl_train.inference_servers.tito.response import build_chat_completion_response
from skyrl.backends.skyrl_train.inference_servers.tito.session import SessionState
from skyrl.backends.skyrl_train.inference_servers.tito.tokenizer_backends import DUMMY_BASE
from skyrl.backends.skyrl_train.inference_servers.tito.tool_parsers.hermes import HermesParser

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MODEL_NAME = "zai-org/GLM-4.7"
RENDERER_NAME = "glm-4.5"

try:
    from transformers import AutoTokenizer
    from renderers import create_renderer

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="transformers/renderers not installed")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return create_renderer(tokenizer, renderer=RENDERER_NAME)


@pytest.fixture(scope="module")
def trajectory() -> Dict[str, Any]:
    with open(FIXTURES_DIR / "trajectory_glm47_synthetic.json") as f:
        return json.load(f)


class TestGLM47TokenizerConsistency:
    """Verify renderer produces correct tokens for GLM-4.7."""

    def test_render_ids_produces_tokens(self, renderer, trajectory):
        msgs = trajectory["messages"][:2]
        ids = renderer.render_ids(msgs, add_generation_prompt=False)
        assert len(ids) > 0

    def test_render_ids_with_gen_prompt(self, renderer, trajectory):
        msgs = trajectory["messages"][:2]
        ids_with = renderer.render_ids(msgs, add_generation_prompt=True)
        ids_without = renderer.render_ids(msgs, add_generation_prompt=False)
        gen_prompt = ids_with[len(ids_without):]
        assert len(gen_prompt) > 0, "Gen prompt should not be empty"

    def test_delta_tokenization(self, renderer, trajectory):
        """Dummy-base delta should produce non-empty tokens."""
        tool_msg = trajectory["messages"][3]
        dummy_ids = renderer.render_ids(DUMMY_BASE, add_generation_prompt=False)
        full_ids = renderer.render_ids(
            DUMMY_BASE + [tool_msg], add_generation_prompt=False,
        )
        delta = full_ids[len(dummy_ids):]
        assert len(delta) > 0, "Delta tokenization should produce tokens"

    def test_delta_is_deterministic(self, renderer, trajectory):
        tool_msg = trajectory["messages"][3]
        dummy_ids = renderer.render_ids(DUMMY_BASE, add_generation_prompt=False)

        full1 = renderer.render_ids(DUMMY_BASE + [tool_msg], add_generation_prompt=False)
        full2 = renderer.render_ids(DUMMY_BASE + [tool_msg], add_generation_prompt=False)
        assert full1[len(dummy_ids):] == full2[len(dummy_ids):]


class TestGLM47Bridge:
    """Test bridge_to_next_turn with GLM-4.5 renderer."""

    def test_bridge_returns_non_none(self, renderer, trajectory):
        msgs = trajectory["messages"][:2]
        prompt_ids = renderer.render_ids(msgs, add_generation_prompt=True)
        result = renderer.bridge_to_next_turn(
            previous_prompt_ids=prompt_ids,
            previous_completion_ids=[100, 200],
            new_messages=[trajectory["messages"][3]],
        )
        assert result is not None, "GLM45Renderer should support bridge"

    def test_bridge_preserves_prefix(self, renderer, trajectory):
        msgs = trajectory["messages"][:2]
        prompt_ids = renderer.render_ids(msgs, add_generation_prompt=True)
        completion_ids = [100, 200, 300]

        result = renderer.bridge_to_next_turn(
            previous_prompt_ids=prompt_ids,
            previous_completion_ids=completion_ids,
            new_messages=[trajectory["messages"][3]],
        )
        assert result is not None
        expected = prompt_ids + completion_ids
        assert result.token_ids[:len(expected)] == expected

    def test_bridge_extends_past_prefix(self, renderer, trajectory):
        msgs = trajectory["messages"][:2]
        prompt_ids = renderer.render_ids(msgs, add_generation_prompt=True)
        completion_ids = [100, 200]

        result = renderer.bridge_to_next_turn(
            previous_prompt_ids=prompt_ids,
            previous_completion_ids=completion_ids,
            new_messages=[trajectory["messages"][3]],
        )
        assert result is not None
        assert len(result.token_ids) > len(prompt_ids) + len(completion_ids)


class TestGLM47SessionAccumulation:
    """Test multi-turn session with GLM-4.7 tokens."""

    def test_full_rollout(self, renderer, trajectory):
        msgs = trajectory["messages"]
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
            dummy_ids = renderer.render_ids(DUMMY_BASE, add_generation_prompt=False)
            full_ids = renderer.render_ids(
                DUMMY_BASE + [msgs[i]], add_generation_prompt=False,
            )
            response_ids = full_ids[len(dummy_ids):]

            session.last_prompt_len = len(session.tokens)
            session.append_response(response_ids)
            session.turn += 1
            turn_count += 1

        assert len(session.tokens) == len(session.loss_mask)
        assert session.turn == turn_count
        assert session.turn == 3  # 3 assistant turns in synthetic trajectory
        assert any(m == 0 for m in session.loss_mask)
        assert any(m == 1 for m in session.loss_mask)
        assert session.loss_mask[0] == 0

    def test_training_split(self, renderer, trajectory):
        """Verify the training split produces valid prompt/response."""
        msgs = trajectory["messages"]
        session = SessionState(model=MODEL_NAME)

        # Quick accumulation
        prompt_ids = renderer.render_ids(msgs[:2], add_generation_prompt=False)
        session.tokens = list(prompt_ids)
        session.loss_mask = [0] * len(prompt_ids)

        dummy_ids = renderer.render_ids(DUMMY_BASE, add_generation_prompt=False)
        full_ids = renderer.render_ids(DUMMY_BASE + [msgs[2]], add_generation_prompt=False)
        response_ids = full_ids[len(dummy_ids):]
        session.append_response(response_ids)

        # Split at first loss_mask=1
        first_gen = next(i for i, m in enumerate(session.loss_mask) if m == 1)
        train_prompt = session.tokens[:first_gen]
        train_response = session.tokens[first_gen:]
        train_mask = session.loss_mask[first_gen:]

        assert len(train_prompt) > 0
        assert len(train_response) > 0
        assert len(train_response) == len(train_mask)
        assert all(m == 1 for m in train_mask)


class TestGLM47PrefixCheck:
    """Prefix checking with GLM-4.7 tokens."""

    def test_same_tokenization_matches(self, renderer):
        msgs = [{"role": "user", "content": "Hello from GLM test"}]
        ids1 = renderer.render_ids(msgs, add_generation_prompt=False)
        ids2 = renderer.render_ids(msgs, add_generation_prompt=False)
        assert _check_prefix("glm_test", 0, ids1, ids2) is True

    def test_different_content_mismatches(self, renderer):
        ids1 = renderer.render_ids(
            [{"role": "user", "content": "Hello"}], add_generation_prompt=False
        )
        ids2 = renderer.render_ids(
            [{"role": "user", "content": "World"}], add_generation_prompt=False
        )
        assert _check_prefix("glm_test", 0, ids1, ids2) is False


class TestGLM47VsQwen3RendererDifference:
    """Verify that GLM and Qwen3 renderers produce different tokens
    for the same messages (different chat templates)."""

    def test_different_tokenization(self, renderer):
        try:
            qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
            qwen_r = create_renderer(qwen_tok, renderer="qwen3")
        except Exception:
            pytest.skip("Qwen3-8B tokenizer not available")

        msgs = [{"role": "user", "content": "Hello"}]
        glm_ids = renderer.render_ids(msgs, add_generation_prompt=True)
        qwen_ids = qwen_r.render_ids(msgs, add_generation_prompt=True)

        # They should produce different token IDs (different vocab + template)
        assert glm_ids != qwen_ids, (
            "GLM and Qwen3 should produce different tokens for same message"
        )
