"""Backwards-compatible re-exports.

The tokenization logic has moved to :mod:`tokenizer_backends`.
This module re-exports the legacy free-function interface for any
external code that imported from the old path.
"""

from .tokenizer_backends import DUMMY_BASE, HttpTokenizerBackend  # noqa: F401

# Legacy free-function wrappers preserved for backward compatibility.
# New code should use ``TokenizerBackend`` instances directly.

__all__ = ["tokenize_messages", "tokenize_delta", "get_gen_prompt_ids", "DUMMY_BASE"]
