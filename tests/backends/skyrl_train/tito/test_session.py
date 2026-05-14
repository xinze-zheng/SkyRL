"""Tests for TITO SessionState."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.session import SessionState


class TestSessionState:
    """Unit tests for SessionState dataclass and methods."""

    def test_default_state(self):
        """New session should have empty tokens, zero turn, zero messages."""
        s = SessionState()
        assert s.tokens == []
        assert s.loss_mask == []
        assert s.turn == 0
        assert s.messages_seen == 0
        assert s.model == ""
        assert s.tools is None

    def test_append_prompt(self):
        """append_prompt should extend tokens with loss_mask=0."""
        s = SessionState()
        s.append_prompt([1, 2, 3])
        assert s.tokens == [1, 2, 3]
        assert s.loss_mask == [0, 0, 0]

    def test_append_response(self):
        """append_response should extend tokens with loss_mask=1."""
        s = SessionState()
        s.append_response([10, 20])
        assert s.tokens == [10, 20]
        assert s.loss_mask == [1, 1]

    def test_prompt_then_response(self):
        """Mixed prompt + response should produce correct masks."""
        s = SessionState()
        s.append_prompt([1, 2])
        s.append_response([3, 4, 5])
        assert s.tokens == [1, 2, 3, 4, 5]
        assert s.loss_mask == [0, 0, 1, 1, 1]

    def test_begin_turn(self):
        """begin_turn should snapshot and return messages_seen."""
        s = SessionState()
        s.messages_seen = 3
        prev = s.begin_turn()
        assert prev == 3
        assert s.prev_messages_seen == 3

        # Updating messages_seen shouldn't affect the snapshot
        s.messages_seen = 7
        assert s.prev_messages_seen == 3

        # Next begin_turn should snapshot the new value
        prev2 = s.begin_turn()
        assert prev2 == 7
        assert s.prev_messages_seen == 7

    def test_to_dict(self):
        """to_dict should include tokens, loss_mask, turn, model."""
        s = SessionState(model="test-model")
        s.append_prompt([1, 2])
        s.append_response([3])
        s.turn = 1
        d = s.to_dict()
        assert d == {
            "tokens": [1, 2, 3],
            "loss_mask": [0, 0, 1],
            "turn": 1,
            "model": "test-model",
        }

    def test_append_empty(self):
        """Appending empty lists should be a no-op."""
        s = SessionState()
        s.append_prompt([])
        s.append_response([])
        assert s.tokens == []
        assert s.loss_mask == []

    def test_multi_turn_accumulation(self):
        """Simulate a 3-turn conversation with correct bookkeeping."""
        s = SessionState(model="qwen3")
        # Turn 0: prompt
        s.append_prompt([10, 20, 30])
        s.messages_seen = 2

        # Turn 0: response
        s.append_response([40, 50])
        s.turn = 1

        # Turn 1: observation
        prev = s.begin_turn()
        assert prev == 2
        s.append_prompt([60, 70])
        s.messages_seen = 4

        # Turn 1: response
        s.append_response([80])
        s.turn = 2

        assert s.tokens == [10, 20, 30, 40, 50, 60, 70, 80]
        assert s.loss_mask == [0, 0, 0, 1, 1, 0, 0, 1]
        assert s.turn == 2
