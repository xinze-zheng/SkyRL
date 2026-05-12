"""TITO proxy configuration.

Defines :class:`TITOConfig`, the configuration dataclass for the
Token-In-Token-Out proxy. Nested under
:class:`~skyrl.train.config.config.InferenceEngineConfig` as the ``tito``
field and constructed automatically by ``build_nested_dataclass``.

CLI example::

    generator.inference_engine.tito.enabled=true
    generator.inference_engine.tito.tool_call_parser=hermes
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TITOConfig:
    """Configuration for the TITO (Token-In-Token-Out) proxy.

    When ``enabled=True``, :func:`build_new_inference_client` inserts a
    lightweight HTTP proxy between the generator and the vLLM router.
    The proxy intercepts ``/v1/chat/completions`` requests, converts them
    to token-level ``/v1/completions`` calls, and bookkeeps exact
    ``(token_ids, loss_mask)`` per session — eliminating training-side
    re-tokenization.
    """

    enabled: bool = False
    """Master switch. When ``True``, the TITO proxy is created and the
    ``RemoteInferenceClient.proxy_url`` is redirected through it."""

    tool_call_parser: str = "hermes"
    """Name of the tool-call parser used to extract structured
    ``tool_calls`` from the raw model output text.

    Supported values:

    - ``"hermes"`` — ``<tool_call>`` XML tags (Qwen3, Hermes models).
    - ``"none"``   — No parsing; full text returned as ``content``.
    """

    log_path: Optional[str] = None
    """Directory for ``tito_proxy.log``. When set, all per-turn
    diagnostics (tokenization, prefix checks, errors) are written to
    this file with line-buffering. When ``None``, diagnostics go to
    stdout."""

    prefix_check: bool = True
    """Run a per-turn sanity check that validates observation-delta
    tokenization against the engine's ``/tokenize`` endpoint.

    Adds one extra ``/tokenize`` call per turn. Recommended during
    development; can be disabled in production for throughput."""

    port: int = 0
    """TCP port for the proxy's HTTP server. ``0`` (default) auto-selects
    an available port via ``get_open_port()``."""
