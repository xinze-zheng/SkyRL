"""Tests for TITO SessionState."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.tito.session import SessionState, TokenPrefixStore


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
        assert d["tokens"] == [1, 2, 3]
        assert d["loss_mask"] == [0, 0, 1]
        assert d["turn"] == 1
        assert d["model"] == "test-model"
        assert d["transitions"] == []
        root = d["input_token_prefix_tree"]["nodes"][0]
        assert root["parent"] is None
        assert root["token"] is None
        assert root["depth"] == 0
        assert root["prefix_hash"].startswith("sha256:")

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

    def test_prefix_store_shares_common_prefix(self):
        """TokenPrefixStore should share cumulative prompt prefixes."""
        store = TokenPrefixStore()
        first = store.add([1, 2, 3])
        second = store.add([1, 2, 3, 4, 5])
        third = store.add([1, 2, 9])

        assert first == [1, 2, 3]
        assert second.to_list() == [1, 2, 3, 4, 5]
        assert third.to_list() == [1, 2, 9]
        assert first.prefix_hash.startswith("sha256:")
        assert store.get_ref_by_hash(second.prefix_hash).to_list() == [1, 2, 3, 4, 5]
        assert store.materialize_hash(third.prefix_hash) == [1, 2, 9]
        # root + shared 1,2 + branch 3 + suffix 4,5 + branch 9
        assert len(store) == 7

    def test_record_transition_compact_input(self):
        """record_transition should store input IDs by prefix-tree reference."""
        s = SessionState(model="m")
        first = s.record_transition(
            step=0,
            input_token_ids=[1, 2, 3],
            output_token_ids=[4, 5],
            output_logprobs=[-0.1, -0.2],
            output_top_logprobs=[{"a": -0.1}, {"b": -0.2}],
            output_text="hello",
            assistant_message={"role": "assistant", "content": "hello"},
            observation_token_ids=[1, 2, 3],
        )
        second = s.record_transition(
            step=1,
            input_token_ids=[1, 2, 3, 4, 5, 6],
            output_token_ids=[7],
            output_logprobs=[-0.3],
            output_top_logprobs=[{"c": -0.3}],
            output_text="next",
            assistant_message={"role": "assistant", "content": "next"},
            observation_token_ids=[6],
        )

        assert first.input_token_ids == [1, 2, 3]
        assert second.input_token_ids.to_list() == [1, 2, 3, 4, 5, 6]
        assert len(s.input_token_store) == 7

        compact = s.to_dict()
        assert "input_token_ids" not in compact["transitions"][0]
        assert compact["transitions"][0]["input_token_ref"] == first.input_token_ids.node_id
        assert compact["transitions"][0]["input_prefix_hash"] == first.input_token_ids.prefix_hash
        assert compact["transitions"][1]["input_token_len"] == 6

        materialized = s.to_dict(include_input_tokens=True)
        assert materialized["transitions"][1]["input_token_ids"] == [1, 2, 3, 4, 5, 6]

    def test_record_transition_rejects_invalid_logprob_shape(self):
        """record_transition should fail when logprob state is misaligned."""
        s = SessionState(model="m")

        with pytest.raises(ValueError, match="output_logprobs"):
            s.record_transition(
                step=0,
                input_token_ids=[1, 2, 3],
                output_token_ids=[4, 5],
                output_logprobs=[-0.1],
                output_top_logprobs=[{"4": -0.1}, {"5": -0.2}],
                output_text="hello",
                assistant_message={"role": "assistant", "content": "hello"},
            )

    def test_record_transition_rejects_invalid_assistant_message(self):
        """record_transition should fail on non-assistant messages."""
        s = SessionState(model="m")

        with pytest.raises(ValueError, match="assistant_message.role"):
            s.record_transition(
                step=0,
                input_token_ids=[1, 2, 3],
                output_token_ids=[4],
                output_logprobs=[-0.1],
                output_top_logprobs=[{"4": -0.1}],
                output_text="hello",
                assistant_message={"role": "user", "content": "hello"},
            )

    def test_to_dict_validates_existing_transition_state(self):
        """to_dict should catch corrupted transition state before exposure."""
        s = SessionState(model="m")
        transition = s.record_transition(
            step=0,
            input_token_ids=[1, 2, 3],
            output_token_ids=[4],
            output_logprobs=[-0.1],
            output_top_logprobs=[{"4": -0.1}],
            output_text="hello",
            assistant_message={"role": "assistant", "content": "hello"},
        )

        transition.step = 2

        with pytest.raises(ValueError, match="transition step mismatch"):
            s.to_dict()
