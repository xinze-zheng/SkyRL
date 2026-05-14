"""TITO proxy configuration.

Defines :class:`TITOConfig`, the configuration dataclass for the
Token-In-Token-Out proxy. Nested under
:class:`~skyrl.train.config.config.InferenceEngineConfig` as the ``tito``
field and constructed automatically by ``build_nested_dataclass``.

Note: ``TITOConfig`` does not inherit ``BaseConfig`` because ``config.py``
already imports ``TITOConfig``, which would create a circular dependency.
``build_nested_dataclass`` handles construction from dicts without needing
the ``from_dict_config`` classmethod.

CLI example::

    generator.inference_engine.tito.enabled=true
    generator.inference_engine.tito.tool_call_parser=hermes
    generator.inference_engine.tito.use_renderer=true
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
    ``loguru`` default output."""

    prefix_check: bool = True
    """Run a per-turn sanity check that validates observation-delta
    tokenization against the engine's ``/tokenize`` endpoint.

    Adds one extra ``/tokenize`` call per turn. Recommended during
    development; can be disabled in production for throughput."""

    port: int = 0
    """TCP port for the proxy's HTTP server. ``0`` (default) auto-selects
    an available port via ``get_open_port()``."""

    use_renderer: bool = False
    """When ``True``, use the ``renderers`` library for local tokenization
    instead of the backend ``/tokenize`` HTTP endpoint. Requires the
    ``renderers`` package to be installed.

    Benefits: eliminates tokenization drift via ``bridge_to_next_turn``,
    removes per-turn ``/tokenize`` round-trips, and provides reliable
    tool-call parsing via special-token boundaries."""

    renderer_name: str = "auto"
    """Renderer name passed to ``renderers.create_renderer()``. ``"auto"``
    auto-detects from the tokenizer's ``name_or_path``. For fine-tuned
    models, pass the base model family explicitly (e.g. ``"qwen3"``)."""
