"""TITO (Token-In-Token-Out) proxy for agentic RL training.

Intercepts ``/v1/chat/completions`` requests, converts them to
token-level ``/v1/completions`` calls, and bookkeeps exact token IDs
and loss masks per session — eliminating training-side re-tokenization.
"""
