# ruff: noqa: E402
"""Integration-facing tests for mini-swe-agent TITO payloads in SkyRL."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.train.mini_swe_agent.mini_swe_generator import MiniSweAgentGenerator
from examples.train.mini_swe_agent.phase3_smoke import _rebuild_tito_arrays


def _tito_blob() -> dict:
    prompt = [1, 2, 3]
    first_output = [10, 11]
    observation = [20, 21, 22]
    second_output = [30, 31, 32]
    tokens = [*prompt, *first_output, *observation, *second_output]
    loss_mask = [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    logprobs = [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4, 0.0]
    return {
        "tokens": tokens,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "prompt_len": len(prompt),
        "transitions": [
            {
                "step": 1,
                "prompt_token_ids": prompt,
                "output_token_ids": first_output,
                "output_logprobs": [-0.1, -0.2],
                "observation_token_ids": [],
            },
            {
                "step": 2,
                "prompt_token_ids": [9999],  # Not used for reconstruction after step 0.
                "output_token_ids": second_output,
                "output_logprobs": [-0.3, -0.4],  # Padded for the final output token.
                "observation_token_ids": observation,
            },
        ],
    }


def test_phase3_rebuilds_arrays_from_fixed_base_transitions_without_prompt_prefix():
    tito = _tito_blob()

    rebuilt_tokens, rebuilt_mask, rebuilt_logprobs = _rebuild_tito_arrays(tito)

    assert rebuilt_tokens == tito["tokens"]
    assert rebuilt_mask == tito["loss_mask"]
    assert rebuilt_logprobs == tito["logprobs"]


def test_skyrl_generator_consumes_tito_payload_without_retokenization():
    tito = _tito_blob()

    response_ids, reward, stop_reason, loss_mask, prompt_ids, logprobs = MiniSweAgentGenerator._build_output_from_tito(
        object(),
        tito,
        max_tokens=32,
        max_input_length=32,
    )

    assert reward == 0.0
    assert stop_reason == "complete"
    assert prompt_ids == [1, 2, 3]
    assert response_ids == [10, 11, 20, 21, 22, 30, 31, 32]
    assert loss_mask == [1, 1, 0, 0, 0, 1, 1, 1]
    assert logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4, 0.0]


def test_skyrl_generator_truncates_tito_response_fields_together():
    tito = _tito_blob()

    response_ids, _reward, stop_reason, loss_mask, prompt_ids, logprobs = MiniSweAgentGenerator._build_output_from_tito(
        object(),
        tito,
        max_tokens=4,
        max_input_length=3,
    )

    assert stop_reason == "length"
    assert prompt_ids == [1, 2, 3]
    assert response_ids == [10, 11, 20, 21]
    assert loss_mask == [1, 1, 0, 0]
    assert logprobs == [-0.1, -0.2, 0.0, 0.0]
