"""Per-session state for TITO token bookkeeping.

Each rollout session is identified by a URL-path session ID (e.g.
``instance_id_rep_id_step``). The proxy maintains a :class:`SessionState`
per session, accumulating token IDs and loss-mask values across turns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    """Accumulated token-level state for a single rollout session.

    Built incrementally by the TITO proxy:

    - **Turn 0**: ``prompt_ids`` appended with ``loss_mask=0``.
    - **Turn N**: observation ``delta_ids`` appended with ``loss_mask=0``,
      then engine ``response_token_ids`` appended with ``loss_mask=1``.

    After the rollout, the generator reads ``(tokens, loss_mask)`` via
    ``GET /session/{id}/data`` — no re-tokenization needed.
    """

    tokens: List[int] = field(default_factory=list)
    """Running list of token IDs: prompt + (response + observation) × N turns.
    Engine response tokens are the exact IDs returned by ``/v1/completions``
    with ``return_token_ids=True``, never re-tokenized."""

    loss_mask: List[int] = field(default_factory=list)
    """Per-token binary mask aligned with ``tokens``.
    ``0`` for prompt / observation tokens (not trained on),
    ``1`` for model-generated response tokens (trained on)."""

    turn: int = 0
    """Number of completed generation turns (incremented after each
    ``/v1/chat/completions`` response is sent)."""

    model: str = ""
    """Model identifier from the first request. Forwarded to
    ``/v1/completions`` and ``/tokenize`` calls."""

    tools: Optional[List[Dict[str, Any]]] = None
    """Tool definitions from the first request. Forwarded to
    ``/tokenize`` so the chat template renders the system message
    identically (e.g. Qwen3 embeds tool descriptions into the system turn
    when ``tools`` is present)."""

    messages_seen: int = 0
    """Number of messages processed so far. Used to compute the delta
    (new messages since last turn) without re-scanning the full history."""

    _prev_messages_seen: int = 0
    """Snapshot of ``messages_seen`` before the current turn's delta was
    applied. Used by the prefix sanity check to locate the observation
    slice in the bookkeeping."""

    def begin_turn(self) -> int:
        """Snapshot ``messages_seen`` at the start of a new turn.

        Must be called before updating ``messages_seen`` for the current
        turn. The returned value is the count of messages processed in
        prior turns, used to compute the delta and by the prefix check.

        Returns:
            Previous ``messages_seen`` count.
        """
        self._prev_messages_seen = self.messages_seen
        return self._prev_messages_seen

    @property
    def prev_messages_seen(self) -> int:
        """Messages-seen snapshot from the last :meth:`begin_turn` call."""
        return self._prev_messages_seen

    def append_prompt(self, token_ids: List[int]) -> None:
        """Append prompt / observation tokens with ``loss_mask=0``."""
        self.tokens.extend(token_ids)
        self.loss_mask.extend([0] * len(token_ids))

    def append_response(self, token_ids: List[int]) -> None:
        """Append model-generated response tokens with ``loss_mask=1``."""
        self.tokens.extend(token_ids)
        self.loss_mask.extend([1] * len(token_ids))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session state for the ``GET /session/{id}/data`` endpoint.

        Returns:
            Dict with ``tokens``, ``loss_mask``, ``turn``, and ``model``.
        """
        return {
            "tokens": self.tokens,
            "loss_mask": self.loss_mask,
            "turn": self.turn,
            "model": self.model,
        }
