"""OpenAI chat-completion response construction for the TITO proxy.

Builds a response dict that matches the OpenAI ``ChatCompletion`` schema,
including optional logprobs conversion from the completions format.

Key detail: ``reasoning_content`` is set to ``None`` explicitly in the
message. This prevents litellm's ``_extract_reasoning_content`` from
auto-parsing ``<think>`` blocks out of ``content`` — preserving thinking
tokens as-is for the agent framework.
"""

import time
import uuid
from typing import Any, Dict, List, Optional


def build_chat_completion_response(
    model: str,
    content: Optional[str],
    tool_calls: Optional[List[Dict[str, Any]]],
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
    logprobs_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct an OpenAI-compatible chat completion response.

    Converts the raw output from ``/v1/completions`` (text + token IDs)
    into the ``ChatCompletion`` format that agent frameworks expect.

    Args:
        model: Model identifier to echo in the response.
        content: Text content for the assistant message (may include
            ``<think>`` blocks). ``None`` when the response is tool-calls
            only.
        tool_calls: Structured tool calls in OpenAI format, or ``None``.
            When present, ``finish_reason`` is overridden to
            ``"tool_calls"``.
        prompt_tokens: Number of prompt tokens (for ``usage``).
        completion_tokens: Number of completion tokens (for ``usage``).
        finish_reason: Engine finish reason (``"stop"``, ``"length"``).
            Overridden to ``"tool_calls"`` when tool calls are present.
        logprobs_data: Raw logprobs from the ``/v1/completions`` response
            (``CompletionLogProbs`` format: ``tokens``,
            ``token_logprobs``, ``top_logprobs``). Converted to the
            ``ChatCompletionLogProbs`` format (``content`` array with
            per-token ``{token, logprob, bytes, top_logprobs}`` entries).
            ``None`` to omit logprobs from the response.

    Returns:
        Dict matching the OpenAI ``ChatCompletion`` JSON schema.
    """
    message: Dict[str, Any] = {
        "role": "assistant",
        "reasoning_content": None,
    }
    if content is not None:
        message["content"] = content
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    # -----------------------------------------------------------------
    # Convert completions logprobs → chat completions format
    # -----------------------------------------------------------------
    choice_logprobs = None
    if logprobs_data:
        tokens = logprobs_data.get("tokens", [])
        token_logprobs = logprobs_data.get("token_logprobs", [])
        top_logprobs_list = logprobs_data.get("top_logprobs", [])
        lp_content: List[Dict[str, Any]] = []
        for i, token_str in enumerate(tokens):
            entry: Dict[str, Any] = {
                "token": token_str,
                "logprob": (
                    token_logprobs[i]
                    if i < len(token_logprobs)
                    and token_logprobs[i] is not None
                    else -9999.0
                ),
                "bytes": list(
                    token_str.encode("utf-8", errors="replace")
                ),
            }
            if i < len(top_logprobs_list) and top_logprobs_list[i]:
                entry["top_logprobs"] = [
                    {
                        "token": t,
                        "logprob": lp,
                        "bytes": list(
                            t.encode("utf-8", errors="replace")
                        ),
                    }
                    for t, lp in top_logprobs_list[i].items()
                ]
            else:
                entry["top_logprobs"] = []
            lp_content.append(entry)
        choice_logprobs = {"content": lp_content}

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": choice_logprobs,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
