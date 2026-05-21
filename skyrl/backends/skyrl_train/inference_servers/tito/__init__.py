"""TITO (Token-In-Token-Out) proxy for agentic RL training.

Intercepts ``/v1/chat/completions`` requests, converts them to
token-level ``/v1/completions`` calls, and bookkeeps exact token IDs
and loss masks per session — eliminating training-side re-tokenization.
"""

from .config import TITOConfig  # noqa: F401
from .proxy import TITOHandler, TITOProxyActor  # noqa: F401
from .session import SessionState, TITOTransition, TokenPrefixStore, TokenSequenceRef  # noqa: F401
from .tokenizer_backends import HttpTokenizerBackend, RendererTokenizerBackend, TokenizerBackend  # noqa: F401
