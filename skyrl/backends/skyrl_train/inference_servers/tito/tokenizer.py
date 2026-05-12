"""Tokenization helpers for the TITO proxy.

All tokenization goes through the backend ``/tokenize`` HTTP endpoint,
which applies the engine's chat template (e.g.
``qwen3_acc_thinking.jinja2``). This avoids loading a local tokenizer in
the proxy and guarantees template consistency with the vLLM engine.

Key functions:

- :func:`tokenize_messages` — tokenize a full message list.
- :func:`tokenize_delta` — tokenize only new messages via the fixed-base
  approach (avoids O(n²) re-tokenization).
- :func:`get_gen_prompt_ids` — extract generation-prompt tokens
  (``<|im_start|>assistant\\n``) via diff.
"""

from typing import Any, Dict, List, Optional

import httpx

# Fixed-base dummy messages for delta tokenization.
# The fixed-base approach tokenizes ``[dummy] + [new_msgs]`` and subtracts
# the dummy prefix to isolate the new-message tokens with correct template
# wrapping.  This is template-agnostic for ChatML-family templates because
# special tokens (``<|im_start|>``, ``<|im_end|>``) act as BPE merge barriers.
#
# References:
#   - https://jybsuper.github.io/posts/multiturn_tokenization/
#   - SkyRL ``encode_messages_subset`` (skyrl/train/generators/utils.py)
#   - SLIME ``_encode_observation_for_generation``
DUMMY_BASE: List[Dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


async def tokenize_messages(
    backend_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    add_generation_prompt: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Tokenize chat messages via the backend ``/tokenize`` endpoint.

    Args:
        backend_url: Base URL of the vLLM router (no ``/v1`` suffix).
        model: Model identifier for the ``/tokenize`` request.
        messages: OpenAI-format message list.
        client: Shared ``httpx.AsyncClient`` for connection pooling.
        add_generation_prompt: Whether to append the generation prompt
            (``<|im_start|>assistant\\n``).  ``False`` for bookkeeping,
            ``True`` when computing the generation-prompt diff.
        tools: Tool definitions to forward.  Required for templates like
            Qwen3 that embed tool descriptions in the system message.

    Returns:
        List of token IDs produced by the engine's chat template.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        payload["tools"] = tools
    resp = await client.post(
        f"{backend_url}/tokenize", json=payload, timeout=30
    )
    resp.raise_for_status()
    return resp.json().get("tokens", [])


async def get_gen_prompt_ids(
    backend_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Extract generation-prompt token IDs via tokenize diff.

    Tokenizes ``messages`` with and without ``add_generation_prompt`` and
    returns the suffix (e.g. ``[<|im_start|>, assistant, \\n]``).

    Args:
        backend_url: Base URL of the vLLM router.
        model: Model identifier.
        messages: Current conversation messages.
        client: Shared ``httpx.AsyncClient``.
        tools: Tool definitions to forward.

    Returns:
        Token IDs for the generation prompt.
    """
    ids_with = await tokenize_messages(
        backend_url, model, messages, client,
        add_generation_prompt=True, tools=tools,
    )
    ids_without = await tokenize_messages(
        backend_url, model, messages, client,
        add_generation_prompt=False, tools=tools,
    )
    return ids_with[len(ids_without):]


async def tokenize_delta(
    backend_url: str,
    model: str,
    new_messages: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Tokenize only new messages using the fixed-base approach.

    Tokenizes ``[DUMMY_BASE] + new_messages``, then subtracts the dummy
    prefix to isolate just the delta tokens with correct template wrapping
    (role headers, ``<|im_end|>\\n``, etc.).

    This is O(1) in conversation history length — only the new messages
    are sent to the tokenizer.

    Args:
        backend_url: Base URL of the vLLM router.
        model: Model identifier.
        new_messages: Only the new messages to tokenize (e.g. a single
            ``tool`` response or ``user`` observation).
        client: Shared ``httpx.AsyncClient``.
        tools: Tool definitions to forward.

    Returns:
        Token IDs for the new messages only.
    """
    dummy_ids = await tokenize_messages(
        backend_url, model, DUMMY_BASE, client,
        add_generation_prompt=False, tools=tools,
    )
    full_ids = await tokenize_messages(
        backend_url, model, DUMMY_BASE + new_messages, client,
        add_generation_prompt=False, tools=tools,
    )
    return full_ids[len(dummy_ids):]
