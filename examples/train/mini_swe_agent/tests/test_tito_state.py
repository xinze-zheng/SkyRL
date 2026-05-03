"""Unit tests for `TITOAgentState`.

These tests are pure-Python: no model, no Ray, no GPU, no vLLM. They
exercise the prefix-equality invariant and slicing behavior that the SkyRL
generator depends on. Runtime: subseconds.

Run with:
    cd SkyRL && uv run pytest examples/train/mini_swe_agent/tests/test_tito_state.py -v
"""

from __future__ import annotations

import pytest

from minisweagent.agents.tito import TITOAgentState, TITOTransition


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _seed_prompt():
    # Pretend tokens for system + user. Concrete values don't matter; only
    # the prefix relationship across steps does.
    return [1, 2, 3, 4, 5]


def _make_transition(step, prev_prompt, prev_output, obs_ids, gen_ids,
                     gen_lp=None):
    """Build a transition whose prompt = prev_prompt + prev_output + obs_ids.

    This is the *invariant* the model class is expected to preserve (vLLM
    extends the prompt when given more messages). The test fixture is
    constructed to honor it; the accumulator is what verifies it.
    """
    new_prompt = list(prev_prompt) + list(prev_output) + list(obs_ids)
    if gen_lp is None:
        gen_lp = [-0.1 * (i + 1) for i in range(len(gen_ids))]
    return TITOTransition(
        step=step,
        prompt_token_ids=new_prompt,
        output_token_ids=list(gen_ids),
        output_logprobs=list(gen_lp),
    )


# ----------------------------------------------------------------------
# Length & alignment
# ----------------------------------------------------------------------
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


def test_step1_appends_only_generation_tokens():
    s = TITOAgentState()
    seed = _seed_prompt()
    s.initialize(seed)
    # First model call uses exactly the seed prompt; no observation delta.
    t1 = TITOTransition(
        step=1, prompt_token_ids=list(seed),
        output_token_ids=[10, 11, 12], output_logprobs=[-0.1, -0.2, -0.3],
    )
    s.absorb_step(t1)

    assert s.tokens == seed + [10, 11, 12]
    assert s.loss_mask == [0, 0, 0, 0, 0, 1, 1, 1]
    assert s.logprobs == [0.0]*5 + [-0.1, -0.2, -0.3]


# ----------------------------------------------------------------------
# Prefix invariant — the central correctness check
# ----------------------------------------------------------------------
def test_two_step_prefix_invariant_holds_with_observation():
    s = TITOAgentState()
    seed = _seed_prompt()
    s.initialize(seed)

    # Step 1: model output [10, 11]
    s.absorb_step(TITOTransition(
        step=1, prompt_token_ids=list(seed),
        output_token_ids=[10, 11], output_logprobs=[-0.1, -0.2],
    ))
    # Step 2: server-rendered prompt = seed + [10,11] + obs[20,21,22]
    obs = [20, 21, 22]
    t2 = _make_transition(
        step=2,
        prev_prompt=seed,
        prev_output=[10, 11],
        obs_ids=obs,
        gen_ids=[30, 31],
    )
    s.absorb_step(t2)

    # All three lists same length, properly labeled.
    n = len(seed) + 2 + 3 + 2  # seed + gen1 + obs + gen2
    assert len(s.tokens) == n
    assert len(s.loss_mask) == n
    assert len(s.logprobs) == n
    assert s.tokens == seed + [10, 11] + obs + [30, 31]
    assert s.loss_mask == [0]*5 + [1, 1] + [0]*3 + [1, 1]


def test_prefix_violation_when_prev_prompt_changes():
    s = TITOAgentState()
    s.initialize([1, 2, 3])
    s.absorb_step(TITOTransition(
        step=1, prompt_token_ids=[1, 2, 3],
        output_token_ids=[10], output_logprobs=[-0.1],
    ))
    # Step 2 prompt does NOT start with previous prompt.
    bogus = TITOTransition(
        step=2, prompt_token_ids=[99, 99, 99, 10],
        output_token_ids=[20], output_logprobs=[-0.2],
    )
    with pytest.raises(ValueError, match="prefix invariant"):
        s.absorb_step(bogus)


def test_prefix_violation_when_prev_output_missing():
    s = TITOAgentState()
    s.initialize([1, 2, 3])
    s.absorb_step(TITOTransition(
        step=1, prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 11], output_logprobs=[-0.1, -0.2],
    ))
    # Step 2 prompt continues from seed but omits the model's prior output.
    bogus = TITOTransition(
        step=2, prompt_token_ids=[1, 2, 3, 99, 99],   # missing [10, 11]
        output_token_ids=[20], output_logprobs=[-0.2],
    )
    with pytest.raises(ValueError, match="prefix invariant"):
        s.absorb_step(bogus)


def test_prefix_violation_when_new_prompt_too_short():
    s = TITOAgentState()
    s.initialize([1, 2, 3])
    s.absorb_step(TITOTransition(
        step=1, prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 11, 12], output_logprobs=[-0.1, -0.2, -0.3],
    ))
    # Truncated prompt — shorter than prev_prompt+prev_output.
    bogus = TITOTransition(
        step=2, prompt_token_ids=[1, 2, 3, 10],
        output_token_ids=[20], output_logprobs=[-0.2],
    )
    with pytest.raises(ValueError, match="prefix invariant"):
        s.absorb_step(bogus)


# ----------------------------------------------------------------------
# Logprob padding (vLLM sometimes omits the stop-token logprob)
# ----------------------------------------------------------------------
def test_logprobs_short_by_one_get_padded():
    s = TITOAgentState()
    s.initialize([1, 2, 3])
    s.absorb_step(TITOTransition(
        step=1, prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 11, 12], output_logprobs=[-0.1, -0.2],  # only 2
    ))
    assert len(s.logprobs) == 6
    assert s.logprobs[-3:] == [-0.1, -0.2, 0.0]   # padded with 0.0


def test_logprobs_too_long_get_truncated():
    s = TITOAgentState()
    s.initialize([1, 2, 3])
    s.absorb_step(TITOTransition(
        step=1, prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 11], output_logprobs=[-0.1, -0.2, -0.3, -0.4],
    ))
    assert s.logprobs[-2:] == [-0.1, -0.2]


# ----------------------------------------------------------------------
# Accessor slicing — what SkyRL's generator reads out
# ----------------------------------------------------------------------
def test_response_slice_matches_loss_mask_and_logprobs():
    s = TITOAgentState()
    seed = _seed_prompt()
    s.initialize(seed)
    s.absorb_step(TITOTransition(
        step=1, prompt_token_ids=list(seed),
        output_token_ids=[10, 11], output_logprobs=[-0.1, -0.2],
    ))
    s.absorb_step(_make_transition(
        step=2, prev_prompt=seed, prev_output=[10, 11],
        obs_ids=[20, 21, 22], gen_ids=[30],
    ))

    # The generator reads these three sequences and feeds them to PPO.
    p = s.prompt_ids()
    r = s.response_ids()
    m = s.response_loss_mask()
    lp = s.response_logprobs()

    assert p == seed
    assert len(r) == len(m) == len(lp)
    # Tokens labeled 1 must be exactly the concatenation of step outputs.
    gens = [tok for tok, mb in zip(r, m) if mb == 1]
    assert gens == [10, 11, 30]


# ----------------------------------------------------------------------
# Serialize() roundtrip — what shows up in trajectory JSONs
# ----------------------------------------------------------------------
def test_serialize_contains_arrays_and_summary():
    s = TITOAgentState()
    s.initialize([1, 2, 3])
    s.absorb_step(TITOTransition(
        step=1, prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 11], output_logprobs=[-0.1, -0.2],
    ))
    blob = s.serialize()["tito"]
    assert blob["n_tokens"] == 5
    assert blob["n_gen_tokens"] == 2
    assert blob["n_obs_tokens"] == 3
    assert blob["n_steps"] == 1
    assert blob["prompt_len"] == 3
    assert blob["response_len"] == 2
    assert blob["tokens"] == [1, 2, 3, 10, 11]
    assert blob["loss_mask"] == [0, 0, 0, 1, 1]
    assert blob["logprobs"] == [0.0, 0.0, 0.0, -0.1, -0.2]
    assert len(blob["transitions"]) == 1
