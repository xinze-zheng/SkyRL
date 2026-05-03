"""Unit tests for fixed-base `TITOAgentState` bookkeeping.

These tests are pure-Python: no model, no Ray, no GPU, no vLLM. They verify
that the saved training arrays are reconstructed from:

    initial prompt + per-turn observation_token_ids + per-turn output_token_ids

without relying on the old cross-turn prompt prefix invariant.

Run with:
    cd SkyRL && uv run pytest examples/train/mini_swe_agent/tests/test_tito_state.py -v
"""

from __future__ import annotations

import pytest
from minisweagent.agents.tito import TITOAgentState, TITOTransition
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class FakeTokenizer:
    eos_token_id = 999
    trailing_newline_id = 10
    generation_prompt_ids = [42, 43]

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is False
        assert kwargs["tools"] == [BASH_TOOL]

        role_ids = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
        token_ids = []
        for message in messages:
            token_ids.append(role_ids[message["role"]])
            token_ids.extend(ord(ch) for ch in (message.get("content") or ""))
            if "tool_call_id" in message:
                token_ids.append(77)
            token_ids.extend([self.eos_token_id, self.trailing_newline_id])
        if kwargs["add_generation_prompt"]:
            token_ids.extend(self.generation_prompt_ids)
        return token_ids


def _seed_prompt() -> list[int]:
    return [1, 2, 3, 4, 5]


def test_initialize_seeds_prompt_with_zero_mask():
    s = TITOAgentState()
    s.initialize(_seed_prompt())
    assert s.tokens == _seed_prompt()
    assert s.loss_mask == [0] * 5
    assert s.logprobs == [0.0] * 5
    assert s.prompt_ids() == _seed_prompt()
    assert s.response_ids() == []


def test_double_initialize_is_rejected():
    s = TITOAgentState()
    s.initialize([1, 2, 3])
    with pytest.raises(RuntimeError):
        s.initialize([4, 5, 6])


def test_first_step_can_seed_prompt_and_append_generation_tokens():
    s = TITOAgentState()
    s.absorb_step(
        TITOTransition(
            step=1,
            prompt_token_ids=_seed_prompt(),
            output_token_ids=[10, 11, 12],
            output_logprobs=[-0.1, -0.2, -0.3],
        )
    )

    assert s.tokens == _seed_prompt() + [10, 11, 12]
    assert s.loss_mask == [0, 0, 0, 0, 0, 1, 1, 1]
    assert s.logprobs == [0.0] * 5 + [-0.1, -0.2, -0.3]


def test_later_steps_use_fixed_base_observation_tokens_not_prompt_prefix():
    tokenizer = FakeTokenizer()
    s = TITOAgentState(tokenizer=tokenizer, chat_template="custom-template")

    s.absorb_step(
        TITOTransition(
            step=1,
            prompt_token_ids=[11, 12],
            output_token_ids=[101, 102],
            output_logprobs=[-0.1, -0.2],
        )
    )
    s.absorb_step(
        TITOTransition(
            step=2,
            prompt_token_ids=[9999],  # Deliberately not a prefix extension.
            output_token_ids=[201],
            output_logprobs=[],
        ),
        new_observation_messages=[
            {
                "role": "tool",
                "content": "obs",
                "tool_call_id": "call_1",
                "extra": {"ignored": True},
            }
        ],
    )

    obs_ids = s.transitions[1].observation_token_ids
    assert obs_ids[0] == tokenizer.trailing_newline_id
    assert obs_ids[-2:] == tokenizer.generation_prompt_ids
    assert s.tokens == [11, 12, 101, 102, *obs_ids, 201]
    assert s.loss_mask == [0, 0, 1, 1, *([0] * len(obs_ids)), 1]
    assert s.logprobs == [0.0, 0.0, -0.1, -0.2, *([0.0] * len(obs_ids)), 0.0]

    observation_call = tokenizer.calls[-1]
    assert observation_call["chat_template"] == "custom-template"
    assert "extra" not in observation_call["messages"][-1]


def test_later_steps_require_tokenizer_for_observation_tokenization():
    s = TITOAgentState()
    s.absorb_step(
        TITOTransition(
            step=1,
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10],
            output_logprobs=[-0.1],
        )
    )

    with pytest.raises(RuntimeError, match="requires a tokenizer"):
        s.absorb_step(
            TITOTransition(
                step=2,
                prompt_token_ids=[999],
                output_token_ids=[20],
                output_logprobs=[-0.2],
            ),
            new_observation_messages=[{"role": "user", "content": "obs"}],
        )


def test_logprobs_short_by_one_get_padded():
    s = TITOAgentState()
    s.absorb_step(
        TITOTransition(
            step=1,
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11, 12],
            output_logprobs=[-0.1, -0.2],
        )
    )
    assert len(s.logprobs) == 6
    assert s.logprobs[-3:] == [-0.1, -0.2, 0.0]


def test_logprobs_too_long_get_truncated():
    s = TITOAgentState()
    s.absorb_step(
        TITOTransition(
            step=1,
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_logprobs=[-0.1, -0.2, -0.3, -0.4],
        )
    )
    assert s.logprobs[-2:] == [-0.1, -0.2]


def test_response_slice_matches_loss_mask_and_logprobs():
    tokenizer = FakeTokenizer()
    s = TITOAgentState(tokenizer=tokenizer)
    seed = _seed_prompt()

    s.absorb_step(
        TITOTransition(
            step=1,
            prompt_token_ids=list(seed),
            output_token_ids=[10, 11],
            output_logprobs=[-0.1, -0.2],
        )
    )
    s.absorb_step(
        TITOTransition(
            step=2,
            prompt_token_ids=[999],
            output_token_ids=[30],
            output_logprobs=[-0.3],
        ),
        new_observation_messages=[{"role": "user", "content": "obs"}],
    )

    p = s.prompt_ids()
    r = s.response_ids()
    m = s.response_loss_mask()
    lp = s.response_logprobs()

    assert p == seed
    assert len(r) == len(m) == len(lp)
    gens = [tok for tok, mb in zip(r, m) if mb == 1]
    assert gens == [10, 11, 30]


def test_serialize_contains_arrays_observation_ids_and_summary():
    tokenizer = FakeTokenizer()
    s = TITOAgentState(tokenizer=tokenizer)
    s.absorb_step(
        TITOTransition(
            step=1,
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_logprobs=[-0.1, -0.2],
        )
    )
    s.absorb_step(
        TITOTransition(
            step=2,
            prompt_token_ids=[999],
            output_token_ids=[20],
            output_logprobs=[-0.3],
        ),
        new_observation_messages=[{"role": "tool", "content": "obs", "tool_call_id": "call_1"}],
    )

    blob = s.serialize()["tito"]
    assert blob["n_tokens"] == len(blob["tokens"])
    assert blob["n_gen_tokens"] == sum(blob["loss_mask"]) == 3
    assert blob["n_obs_tokens"] == len(blob["tokens"]) - 3
    assert blob["n_steps"] == 2
    assert blob["prompt_len"] == 3
    assert blob["response_len"] == len(blob["tokens"]) - 3
    assert len(blob["tokens"]) == len(blob["loss_mask"]) == len(blob["logprobs"])
    assert blob["transitions"][0]["observation_token_ids"] == []
    assert blob["transitions"][1]["observation_token_ids"]
