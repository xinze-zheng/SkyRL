"""Per-session state for TITO token bookkeeping.

Each rollout session is identified by a URL-path session ID (e.g.
``instance_id_rep_id_step``). The proxy maintains a :class:`SessionState`
per session, accumulating token IDs and loss-mask values across turns.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union


@dataclass
class _PrefixNode:
    """One node in the shared input-token prefix tree."""

    parent: Optional[int]
    token: Optional[int]
    depth: int
    prefix_hash: str
    children: Dict[int, int] = field(default_factory=dict)


class TokenPrefixStore:
    """Compact storage for repeated cumulative prompt token sequences.

    Multi-turn agent prompts are usually cumulative: turn N starts with the
    full token stream used by turn N-1, plus the previous response and new
    observation.  Storing every transition's full ``input_token_ids`` would
    duplicate that prefix each turn.  This store interns token sequences in a
    trie and lets each transition keep only a leaf reference.
    """

    _HASH_VERSION = b"tito-prefix-store-v1"

    def __init__(self) -> None:
        root_hash = self._hash_root()
        self._nodes: List[_PrefixNode] = [
            _PrefixNode(parent=None, token=None, depth=0, prefix_hash=root_hash)
        ]
        self._hash_to_node_id: Dict[str, int] = {root_hash: 0}

    @classmethod
    def _hash_root(cls) -> str:
        return "sha256:" + hashlib.sha256(cls._HASH_VERSION).hexdigest()

    @staticmethod
    def _hash_child(parent_hash: str, token: int) -> str:
        token_bytes = int(token).to_bytes(8, byteorder="big", signed=True)
        payload = parent_hash.encode("utf-8") + b"\0" + token_bytes
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def add(self, token_ids: Sequence[int]) -> "TokenSequenceRef":
        """Intern *token_ids* and return a lightweight sequence reference."""
        node_id = 0
        for token in token_ids:
            node = self._nodes[node_id]
            child_id = node.children.get(token)
            if child_id is None:
                child_id = len(self._nodes)
                node.children[token] = child_id
                child_hash = self._hash_child(node.prefix_hash, token)
                self._nodes.append(
                    _PrefixNode(
                        parent=node_id,
                        token=token,
                        depth=node.depth + 1,
                        prefix_hash=child_hash,
                    )
                )
                self._hash_to_node_id[child_hash] = child_id
            node_id = child_id
        return TokenSequenceRef(store=self, node_id=node_id)

    def materialize(self, node_id: int) -> List[int]:
        """Return the token sequence represented by *node_id*."""
        if node_id < 0 or node_id >= len(self._nodes):
            raise IndexError(f"Unknown token prefix node id: {node_id}")

        tokens: List[int] = []
        current_id = node_id
        while current_id != 0:
            node = self._nodes[current_id]
            if node.token is None or node.parent is None:
                break
            tokens.append(node.token)
            current_id = node.parent
        tokens.reverse()
        return tokens

    def get_ref_by_hash(self, prefix_hash: str) -> "TokenSequenceRef":
        """Return a token sequence reference by stable prefix hash."""
        node_id = self._hash_to_node_id.get(prefix_hash)
        if node_id is None:
            raise KeyError(f"Unknown token prefix hash: {prefix_hash}")
        return TokenSequenceRef(store=self, node_id=node_id)

    def materialize_hash(self, prefix_hash: str) -> List[int]:
        """Return the token sequence represented by *prefix_hash*."""
        return self.get_ref_by_hash(prefix_hash).to_list()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the prefix tree without child maps.

        Consumers can reconstruct any transition input by walking parent links
        from ``input_token_ref`` back to the root.
        """
        return {
            "nodes": [
                {
                    "parent": node.parent,
                    "token": node.token,
                    "depth": node.depth,
                    "prefix_hash": node.prefix_hash,
                }
                for node in self._nodes
            ]
        }

    def __len__(self) -> int:
        return len(self._nodes)


@dataclass
class TokenSequenceRef:
    """List-like reference to an interned token sequence."""

    store: TokenPrefixStore = field(repr=False, compare=False)
    node_id: int

    def to_list(self) -> List[int]:
        return self.store.materialize(self.node_id)

    @property
    def prefix_hash(self) -> str:
        """Stable hash identifying the full token prefix."""
        return self.store._nodes[self.node_id].prefix_hash

    def __len__(self) -> int:
        return self.store._nodes[self.node_id].depth

    def __iter__(self) -> Iterator[int]:
        return iter(self.to_list())

    def __getitem__(self, index: Union[int, slice]) -> Union[int, List[int]]:
        return self.to_list()[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TokenSequenceRef):
            return self.to_list() == other.to_list()
        if isinstance(other, list):
            return self.to_list() == other
        return False


@dataclass
class TITOTransition:
    """One token-in/token-out request in the agent loop.

    ``input_token_ids`` is the full prompt sent to ``/v1/completions`` for
    this step. It is stored as a prefix-tree reference to avoid duplicating
    cumulative prompt prefixes across turns.
    """

    step: int
    input_token_ids: TokenSequenceRef
    output_token_ids: List[int]
    output_logprobs: List[float]
    output_top_logprobs: List[Dict[str, float]]
    output_text: str
    assistant_message: Dict[str, Any]
    observation_token_ids: Optional[List[int]] = None

    @staticmethod
    def _validate_token_ids(name: str, token_ids: Sequence[int]) -> None:
        for index, token_id in enumerate(token_ids):
            if not isinstance(token_id, int):
                raise ValueError(f"{name}[{index}] must be int, got {type(token_id).__name__}")

    def validate(self) -> None:
        """Validate transition invariants before runtime exposure."""
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"transition step must be a non-negative int, got {self.step!r}")

        if not isinstance(self.input_token_ids, TokenSequenceRef):
            raise ValueError("input_token_ids must be a TokenSequenceRef")
        if self.input_token_ids.node_id < 0 or self.input_token_ids.node_id >= len(self.input_token_ids.store):
            raise ValueError(f"unknown input_token_ref: {self.input_token_ids.node_id}")

        materialized_input = self.input_token_ids.to_list()
        self._validate_token_ids("input_token_ids", materialized_input)
        try:
            by_hash = self.input_token_ids.store.get_ref_by_hash(self.input_token_ids.prefix_hash)
        except KeyError as exc:
            raise ValueError(
                f"unknown input_prefix_hash: {self.input_token_ids.prefix_hash}"
            ) from exc
        if by_hash.node_id != self.input_token_ids.node_id:
            raise ValueError(
                "input_prefix_hash does not resolve to input_token_ref "
                f"{self.input_token_ids.node_id}"
            )

        self._validate_token_ids("output_token_ids", self.output_token_ids)
        if self.observation_token_ids is not None:
            self._validate_token_ids("observation_token_ids", self.observation_token_ids)

        if not isinstance(self.output_text, str):
            raise ValueError("output_text must be a string")
        if not isinstance(self.assistant_message, dict):
            raise ValueError("assistant_message must be a dict")
        if self.assistant_message.get("role") != "assistant":
            raise ValueError("assistant_message.role must be 'assistant'")

        output_len = len(self.output_token_ids)
        if self.output_logprobs and len(self.output_logprobs) != output_len:
            raise ValueError(
                "output_logprobs must be empty or match output_token_ids length "
                f"({len(self.output_logprobs)} != {output_len})"
            )
        for index, logprob in enumerate(self.output_logprobs):
            if not isinstance(logprob, (int, float)):
                raise ValueError(
                    f"output_logprobs[{index}] must be numeric, got {type(logprob).__name__}"
                )

        if self.output_top_logprobs and len(self.output_top_logprobs) != output_len:
            raise ValueError(
                "output_top_logprobs must be empty or match output_token_ids length "
                f"({len(self.output_top_logprobs)} != {output_len})"
            )
        for index, top_logprobs in enumerate(self.output_top_logprobs):
            if not isinstance(top_logprobs, dict):
                raise ValueError(
                    f"output_top_logprobs[{index}] must be a dict, got {type(top_logprobs).__name__}"
                )
            for token, logprob in top_logprobs.items():
                if not isinstance(token, str):
                    raise ValueError(f"output_top_logprobs[{index}] keys must be strings")
                if not isinstance(logprob, (int, float)):
                    raise ValueError(
                        f"output_top_logprobs[{index}][{token!r}] must be numeric"
                    )

    def to_dict(self, *, include_input_tokens: bool = False) -> Dict[str, Any]:
        """Serialize the transition.

        By default the full input prompt is represented compactly by
        ``input_token_ref``. Set ``include_input_tokens=True`` for debugging
        or consumers that do not want to materialize from the prefix tree.
        """
        self.validate()
        data: Dict[str, Any] = {
            "step": self.step,
            "input_token_ref": self.input_token_ids.node_id,
            "input_prefix_hash": self.input_token_ids.prefix_hash,
            "input_token_len": len(self.input_token_ids),
            "output_token_ids": self.output_token_ids,
            "output_logprobs": self.output_logprobs,
            "output_top_logprobs": self.output_top_logprobs,
            "output_text": self.output_text,
            "assistant_message": self.assistant_message,
            "observation_token_ids": self.observation_token_ids,
        }
        if include_input_tokens:
            data["input_token_ids"] = self.input_token_ids.to_list()
        return data


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

    last_prompt_len: int = 0
    """Length of the prompt token IDs (excluding gen prompt) at the end of
    the last completed turn. Used by ``bridge_to_next_turn`` to split
    ``session.tokens`` into ``previous_prompt_ids`` and
    ``previous_completion_ids``."""

    input_token_store: TokenPrefixStore = field(default_factory=TokenPrefixStore)
    """Shared prefix tree for transition input prompts."""

    transitions: List[TITOTransition] = field(default_factory=list)
    """Per-request token-in/token-out records for this rollout session."""

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

    def reset_from_full_render(self, token_ids: List[int], loss_mask: List[int]) -> None:
        """Replace all token state with a fresh full re-render.

        Called when ``bridge_to_next_turn`` returns ``None`` and the
        handler falls back to re-rendering the entire conversation.
        The new token sequence becomes the baseline for future turns.

        Args:
            token_ids: Fully re-rendered token IDs for the conversation.
            loss_mask: Corresponding loss mask (0 for all, since
                re-rendered tokens lose per-token attribution).
        """
        self.tokens = list(token_ids)
        self.loss_mask = list(loss_mask)

    def record_transition(
        self,
        *,
        step: int,
        input_token_ids: Sequence[int],
        output_token_ids: Sequence[int],
        output_logprobs: Optional[Sequence[float]],
        output_top_logprobs: Optional[Sequence[Dict[str, float]]],
        output_text: str,
        assistant_message: Dict[str, Any],
        observation_token_ids: Optional[Sequence[int]] = None,
    ) -> TITOTransition:
        """Append a compact transition record and return it."""
        transition = TITOTransition(
            step=step,
            input_token_ids=self.input_token_store.add(input_token_ids),
            output_token_ids=list(output_token_ids),
            output_logprobs=list(output_logprobs or []),
            output_top_logprobs=[dict(item) for item in (output_top_logprobs or [])],
            output_text=output_text,
            assistant_message=dict(assistant_message),
            observation_token_ids=(
                list(observation_token_ids) if observation_token_ids is not None else None
            ),
        )
        transition.validate()
        self.transitions.append(transition)
        return transition

    def validate_transitions(self) -> None:
        """Validate all recorded transition records."""
        for index, transition in enumerate(self.transitions):
            transition.validate()
            if transition.step != index:
                raise ValueError(
                    f"transition step mismatch at index {index}: got step {transition.step}"
                )

    def to_dict(self, *, include_input_tokens: bool = False) -> Dict[str, Any]:
        """Serialize session state for the ``GET /session/{id}/data`` endpoint.

        Returns:
            Dict with the legacy ``tokens`` / ``loss_mask`` view plus compact
            transition records and their shared input-token prefix tree.
        """
        self.validate_transitions()
        return {
            "tokens": self.tokens,
            "loss_mask": self.loss_mask,
            "turn": self.turn,
            "model": self.model,
            "transitions": [
                transition.to_dict(include_input_tokens=include_input_tokens)
                for transition in self.transitions
            ],
            "input_token_prefix_tree": self.input_token_store.to_dict(),
        }
