"""Tokenizer backend protocol for the TITO proxy.

Defines :class:`TokenizerBackend`, the interface that decouples the TITO
proxy from any specific tokenization strategy. Two implementations ship:

- :class:`HttpTokenizerBackend` — delegates to the vLLM engine's
  ``/tokenize`` HTTP endpoint (original behaviour).
- :class:`RendererTokenizerBackend` — uses the ``renderers`` library for
  local, drift-free tokenization with ``bridge_to_next_turn``.

The proxy selects a backend at startup based on
:attr:`TITOConfig.use_renderer`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import httpx
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenizerBackend(Protocol):
    """Strategy interface for message → token-ID conversion.

    Implementations must be async-safe. The proxy creates one backend
    instance at app startup and shares it across all sessions.
    """

    async def tokenize_messages(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Tokenize a full message list.

        Args:
            model: Model identifier.
            messages: OpenAI-format message list.
            add_generation_prompt: Whether to append the generation
                prompt (``<|im_start|>assistant\\n``).
            tools: Tool definitions to forward.

        Returns:
            List of token IDs.
        """
        ...

    async def tokenize_delta(
        self,
        model: str,
        new_messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Tokenize only new (observation) messages.

        Args:
            model: Model identifier.
            new_messages: New messages to tokenize.
            tools: Tool definitions to forward.

        Returns:
            Token IDs for the new messages only.
        """
        ...

    async def get_gen_prompt_ids(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Extract generation-prompt token IDs.

        Returns the suffix tokens added by ``add_generation_prompt=True``
        (e.g. ``<|im_start|>assistant\\n``).

        Args:
            model: Model identifier.
            messages: Current conversation messages.
            tools: Tool definitions to forward.

        Returns:
            Token IDs for the generation prompt.
        """
        ...


# ---------------------------------------------------------------------------
# HTTP backend (original /tokenize approach)
# ---------------------------------------------------------------------------

# Fixed-base dummy messages for delta tokenization.
# The fixed-base approach tokenizes ``[dummy] + [new_msgs]`` and subtracts
# the dummy prefix to isolate the new-message tokens with correct template
# wrapping.  This is template-agnostic for ChatML-family templates because
# special tokens (``<|im_start|>``, ``<|im_end|>``) act as BPE merge barriers.
#
# References:
#   - https://jybsuper.github.io/posts/multiturn_tokenization/
#   - SkyRL ``encode_messages_subset`` (skyrl/train/generators/utils.py)
DUMMY_BASE: List[Dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


class HttpTokenizerBackend:
    """Tokenizer backend that delegates to the vLLM ``/tokenize`` endpoint.

    All tokenization goes through the engine, guaranteeing template
    consistency. Delta tokenization uses the fixed-base approach
    (``DUMMY_BASE + new_messages`` minus ``DUMMY_BASE``) to avoid
    O(n²) re-tokenization.

    Args:
        backend_url: Base URL of the vLLM router (no ``/v1`` suffix).
        client: Shared ``httpx.AsyncClient`` for connection pooling.
    """

    def __init__(self, backend_url: str, client: httpx.AsyncClient) -> None:
        self._backend_url = backend_url
        self._client = client

    async def _tokenize(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Call the backend ``/tokenize`` endpoint."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            payload["tools"] = tools
        resp = await self._client.post(
            f"{self._backend_url}/tokenize", json=payload, timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("tokens", [])

    async def tokenize_messages(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Tokenize chat messages via the backend ``/tokenize`` endpoint."""
        return await self._tokenize(
            model, messages, add_generation_prompt=add_generation_prompt, tools=tools,
        )

    async def tokenize_delta(
        self,
        model: str,
        new_messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Tokenize only new messages using the fixed-base approach."""
        dummy_ids = await self._tokenize(
            model, DUMMY_BASE, add_generation_prompt=False, tools=tools,
        )
        full_ids = await self._tokenize(
            model, DUMMY_BASE + new_messages, add_generation_prompt=False, tools=tools,
        )
        return full_ids[len(dummy_ids):]

    async def get_gen_prompt_ids(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Extract generation-prompt token IDs via tokenize diff."""
        ids_with = await self._tokenize(
            model, messages, add_generation_prompt=True, tools=tools,
        )
        ids_without = await self._tokenize(
            model, messages, add_generation_prompt=False, tools=tools,
        )
        return ids_with[len(ids_without):]


# ---------------------------------------------------------------------------
# Renderer backend (renderers library)
# ---------------------------------------------------------------------------


class RendererTokenizerBackend:
    """Tokenizer backend using the ``renderers`` library.

    Performs all tokenization locally (no HTTP round-trips) and supports
    :meth:`bridge_to_next_turn` for drift-free multi-turn extension.

    Falls back to :class:`HttpTokenizerBackend` when ``bridge_to_next_turn``
    returns ``None`` (e.g. ``DefaultRenderer`` for unsupported models).

    Args:
        renderer: A ``renderers.Renderer`` instance.
        http_fallback: ``HttpTokenizerBackend`` for fallback when the
            renderer cannot bridge.
    """

    def __init__(self, renderer: Any, http_fallback: HttpTokenizerBackend) -> None:
        self._renderer = renderer
        self._http_fallback = http_fallback

    @staticmethod
    def create(
        model_name: str,
        renderer_name: str,
        backend_url: str,
        client: httpx.AsyncClient,
    ) -> "RendererTokenizerBackend":
        """Factory that creates a renderer and wraps it.

        Args:
            model_name: HuggingFace model name or path.
            renderer_name: Renderer name (``"auto"`` for auto-detect).
            backend_url: Backend URL for the HTTP fallback.
            client: Shared HTTP client for the fallback backend.

        Returns:
            Configured ``RendererTokenizerBackend``.

        Raises:
            ImportError: If the ``renderers`` package is not installed.
        """
        try:
            from renderers import create_renderer
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "The 'renderers' package is required when tito.use_renderer=True. "
                "Install it with: uv pip install renderers"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        renderer = create_renderer(tokenizer, renderer=renderer_name)
        http_fallback = HttpTokenizerBackend(backend_url, client)
        logger.info(
            f"RendererTokenizerBackend: created {type(renderer).__name__} "
            f"for {model_name} (renderer={renderer_name})"
        )
        return RendererTokenizerBackend(renderer, http_fallback)

    def _to_renderer_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-format messages to renderer-compatible format.

        The ``renderers`` library expects messages as dicts with ``role``
        and ``content`` keys. Tool-call messages need ``tool_calls`` in
        the format the renderer expects.
        """
        # renderers accepts standard OpenAI message dicts directly
        return messages

    def _to_renderer_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Any]]:
        """Convert OpenAI tool definitions to renderer format."""
        # renderers accepts standard OpenAI tool spec dicts
        return tools

    async def tokenize_messages(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Tokenize messages locally using the renderer."""
        r_messages = self._to_renderer_messages(messages)
        r_tools = self._to_renderer_tools(tools)
        return self._renderer.render_ids(
            r_messages,
            tools=r_tools,
            add_generation_prompt=add_generation_prompt,
        )

    async def tokenize_delta(
        self,
        model: str,
        new_messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Tokenize delta messages locally using the renderer.

        Uses the same fixed-base approach as ``HttpTokenizerBackend``
        but with local tokenization (no HTTP calls).
        """
        r_tools = self._to_renderer_tools(tools)
        dummy_ids = self._renderer.render_ids(
            DUMMY_BASE, tools=r_tools, add_generation_prompt=False,
        )
        full_ids = self._renderer.render_ids(
            DUMMY_BASE + self._to_renderer_messages(new_messages),
            tools=r_tools,
            add_generation_prompt=False,
        )
        return full_ids[len(dummy_ids):]

    async def get_gen_prompt_ids(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Extract generation-prompt token IDs locally."""
        r_messages = self._to_renderer_messages(messages)
        r_tools = self._to_renderer_tools(tools)
        ids_with = self._renderer.render_ids(
            r_messages, tools=r_tools, add_generation_prompt=True,
        )
        ids_without = self._renderer.render_ids(
            r_messages, tools=r_tools, add_generation_prompt=False,
        )
        return ids_with[len(ids_without):]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: List[int],
        previous_completion_ids: List[int],
        new_messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[List[int]]:
        """Extend the token stream using ``bridge_to_next_turn``.

        Returns the full next-turn prompt IDs (prev_prompt + prev_completion
        + new observation + gen prompt) if the renderer can prove
        correctness, otherwise ``None``.

        Args:
            previous_prompt_ids: Token IDs of the previous prompt.
            previous_completion_ids: Token IDs of the model's completion.
            new_messages: New observation/tool messages.
            tools: Tool definitions.

        Returns:
            Full next-turn prompt IDs, or ``None`` if bridge is unsafe.
        """
        r_messages = self._to_renderer_messages(new_messages)
        r_tools = self._to_renderer_tools(tools)
        result = self._renderer.bridge_to_next_turn(
            previous_prompt_ids=previous_prompt_ids,
            previous_completion_ids=previous_completion_ids,
            new_messages=r_messages,
            tools=r_tools,
        )
        if result is None:
            return None
        return result.token_ids

    def parse_response(self, token_ids: List[int]) -> Dict[str, Any]:
        """Parse completion token IDs into structured content + tool calls.

        Uses the renderer's special-token-aware parsing, which is more
        reliable than regex-based text parsing.

        Args:
            token_ids: Raw completion token IDs from the engine.

        Returns:
            Dict with ``content`` and ``tool_calls`` keys.
        """
        parsed = self._renderer.parse_response(token_ids)
        tool_calls = None
        if parsed.tool_calls:
            import json
            import uuid

            tool_calls = []
            for tc in parsed.tool_calls:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", tc.get("function", {}).get("name", "")),
                        "arguments": (
                            json.dumps(tc.get("arguments", tc.get("function", {}).get("arguments", {})),
                                       ensure_ascii=False)
                            if not isinstance(tc.get("arguments", tc.get("function", {}).get("arguments", "")), str)
                            else tc.get("arguments", tc.get("function", {}).get("arguments", ""))
                        ),
                    },
                })

        content = parsed.content
        if parsed.reasoning_content and content:
            content = f"<think>{parsed.reasoning_content}</think>\n\n{content}"
        elif parsed.reasoning_content:
            content = f"<think>{parsed.reasoning_content}</think>"

        return {"content": content, "tool_calls": tool_calls}
