"""TITO Proxy — HTTP server for token-in-token-out agentic generation.

This module provides :class:`TITOProxyActor`, a lightweight HTTP proxy
that intercepts OpenAI ``/v1/chat/completions`` requests from agent
frameworks, converts them to token-level ``/v1/completions`` calls against
the vLLM backend, and bookkeeps exact ``(token_ids, loss_mask)`` per
rollout session.

Architecture::

    Agent (litellm / any OpenAI client)
      │  POST /session/{id}/v1/chat/completions
      ▼
    TITOProxyActor (this module)
      │  TokenizerBackend.tokenize_*   ← tokenize observations
      │  POST {backend}/v1/completions ← generate with token IDs
      ▼
    VLLMRouter → vLLM engines

The proxy runs a FastAPI/uvicorn server in a daemon thread. It can be
used standalone or wrapped as a Ray actor.

Usage (standalone)::

    proxy = TITOProxyActor(backend_url="http://...:49999", config=cfg)
    proxy_url = proxy.start()   # blocks until healthy
    # agent sends to proxy_url/session/{id}/v1/chat/completions

Usage (Ray actor)::

    Actor = ray.remote(TITOProxyActor)
    ref = Actor.remote(backend_url="http://...:49999", config=cfg)
    proxy_url = ray.get(ref.start.remote())
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from typing import Any, Dict, List, Optional
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import TITOConfig
from .response import build_chat_completion_response
from .session import SessionState
from .tokenizer_backends import HttpTokenizerBackend, RendererTokenizerBackend, TokenizerBackend
from .tool_parsers import get_tool_parser

logger = logging.getLogger(__name__)

def _check_prefix(
    session_id: str,
    turn: int,
    prev_tokens: List[int],
    cur_tokens: List[int],
) -> bool:
    """Verify that *cur_tokens* starts with *prev_tokens* (prefix match).

    Used by the per-turn sanity check to detect tokenization drift.
    Logs a detailed mismatch report (with a 10-token window around the
    first divergence) when the prefix doesn't match.

    Args:
        session_id: Session identifier (for log messages).
        turn: Current turn number (for log messages).
        prev_tokens: Expected prefix token IDs.
        cur_tokens: Actual token IDs to check against.

    Returns:
        ``True`` if *cur_tokens* starts with *prev_tokens*, ``False``
        otherwise.
    """
    prefix_len = len(prev_tokens)
    if prefix_len == 0:
        return True
    cur_prefix = cur_tokens[:prefix_len]
    if cur_prefix == prev_tokens:
        return True
    first_diff = next(
        (i for i in range(min(prefix_len, len(cur_prefix))) if cur_prefix[i] != prev_tokens[i]),
        min(prefix_len, len(cur_prefix)),
    )
    window = 10
    start = max(0, first_diff - window)
    end = min(max(prefix_len, len(cur_prefix)), first_diff + window)
    logger.warning(
        f"WARNING: session={session_id} turn={turn}: "
        f"MISMATCH at token {first_diff}/{prefix_len} (cur total={len(cur_tokens)})\n"
        f"  prev[{start}:{end}] = {prev_tokens[start:end]}\n"
        f"  cur [{start}:{end}] = {cur_tokens[start:end]}"
    )
    return False


class TITOHandler:
    """Core request handler for TITO chat completions.

    Encapsulates the tokenize → generate → bookkeep → parse → prefix-check
    pipeline.

    Args:
        backend_url: Base URL of the vLLM router.
        config: TITO configuration.
        tokenizer_backend: Strategy for message tokenization.
        sessions: Shared session state dict.
        http_client: Shared ``httpx.AsyncClient`` for engine calls.
    """

    def __init__(
        self,
        backend_url: str,
        config: TITOConfig,
        tokenizer_backend: TokenizerBackend,
        sessions: Dict[str, SessionState],
        http_client: httpx.AsyncClient,
    ) -> None:
        self._backend_url = backend_url
        self._config = config
        self._tokenizer = tokenizer_backend
        self._sessions = sessions
        self._client = http_client
        self._tool_parser = get_tool_parser(config.tool_call_parser)
        self._renderer_backend: Optional[RendererTokenizerBackend] = (
            tokenizer_backend if isinstance(tokenizer_backend, RendererTokenizerBackend) else None
        )

    async def handle_chat_completion(
        self, session_id: str, req_json: Dict[str, Any]
    ) -> JSONResponse:
        """Process a single ``/v1/chat/completions`` request.

        Args:
            session_id: Unique rollout session identifier.
            req_json: Parsed JSON body of the chat completion request.

        Returns:
            ``JSONResponse`` with OpenAI-compatible chat completion.
        """
        messages: List[Dict[str, Any]] = req_json.get("messages", [])
        model = req_json.get("model", "default")
        tools: Optional[List[Dict[str, Any]]] = req_json.get("tools")

        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(model=model, tools=tools)

        session = self._sessions[session_id]
        turn = session.turn
        logger.debug(
            f"session={session_id} turn={turn} "
            f"n_msgs={len(messages)} roles={[m.get('role', '?') for m in messages]}"
        )

        try:
            # Tokenize input
            if turn == 0:
                prompt_ids = await self._tokenizer.tokenize_messages(
                    model, messages, add_generation_prompt=False, tools=tools,
                )
                session.tokens = list(prompt_ids)
                session.loss_mask = [0] * len(prompt_ids)
                session.messages_seen = len(messages)
            else:
                prev_seen = session.begin_turn()
                new_messages = messages[prev_seen:]
                obs_start = 0
                # Skip new assistant messages at the start (already bookkeept)
                while obs_start < len(new_messages) and new_messages[obs_start].get("role") == "assistant":
                    obs_start += 1
                obs_messages = new_messages[obs_start:]

                if obs_messages:
                    bridged = False

                    # Strategy 1: bridge_to_next_turn (renderer only, drift-free)
                    if self._renderer_backend is not None:
                        previous_prompt_ids = session.tokens[:session.last_prompt_len]
                        previous_completion_ids = session.tokens[session.last_prompt_len:]
                        bridge_result = self._renderer_backend.bridge_to_next_turn(
                            previous_prompt_ids, previous_completion_ids,
                            obs_messages, tools=tools,
                        )
                        if bridge_result is not None:
                            # Bridge succeeded: the result is the full next prompt
                            # (prev_prompt + prev_completion + new_obs + gen_prompt).
                            # Extract the tokens without the gen prompt suffix by
                            # comparing to a no-gen-prompt render length.
                            gen_prompt_len = len(await self._tokenizer.get_gen_prompt_ids(
                                model, messages, tools=tools,
                            ))
                            new_tokens = bridge_result[:len(bridge_result) - gen_prompt_len]
                            # Rebuild loss mask: preserve old mask, append 0s for new obs
                            new_obs_len = len(new_tokens) - len(session.tokens)
                            if new_obs_len >= 0:
                                session.tokens = list(new_tokens)
                                session.loss_mask.extend([0] * new_obs_len)
                                bridged = True
                                logger.debug(
                                    f"session={session_id} turn={turn}: "
                                    f"bridge OK, +{new_obs_len} obs tokens"
                                )

                        if not bridged:
                            # Strategy 2: full re-render (state reset)
                            logger.info(
                                f"session={session_id} turn={turn}: "
                                f"bridge returned None, falling back to full re-render"
                            )
                            full_ids = await self._tokenizer.tokenize_messages(
                                model, messages, add_generation_prompt=False, tools=tools,
                            )
                            # Rebuild loss mask: we lose per-token attribution, but
                            # we can recover it from the previous mask for the prefix
                            # that matches, and mark everything else as 0.
                            old_len = min(len(session.loss_mask), len(full_ids))
                            new_mask = session.loss_mask[:old_len] + [0] * (len(full_ids) - old_len)
                            session.reset_from_full_render(full_ids, new_mask)
                            bridged = True
                    else:
                        # Strategy 3: dummy-base delta (no renderer available)
                        delta_ids = await self._tokenizer.tokenize_delta(
                            model, obs_messages, tools=tools,
                        )
                        session.append_prompt(delta_ids)
                        logger.debug(
                            f"session={session_id} turn={turn}: "
                            f"obs {len(obs_messages)} msgs → {len(delta_ids)} tokens "
                            f"(skipped {obs_start} assistant msgs)"
                        )

                session.messages_seen = len(messages)

            # Generation prompt 
            gen_prompt_ids = await self._tokenizer.get_gen_prompt_ids(
                model, messages, tools=tools,
            )
            full_prompt_ids = session.tokens + gen_prompt_ids

            #  /v1/completions with token IDs
            completion_params: Dict[str, Any] = {
                "model": model,
                "prompt": full_prompt_ids,
                "logprobs": 1,
                "echo": False,
                "return_token_ids": True,
                "skip_special_tokens": False,
            }
            for key in (
                "max_tokens", "temperature", "top_p", "top_k",
                "stop", "frequency_penalty", "presence_penalty", "seed",
            ):
                if key in req_json:
                    completion_params[key] = req_json[key]
            if "max_completion_tokens" in req_json and "max_tokens" not in completion_params:
                completion_params["max_tokens"] = req_json["max_completion_tokens"]

            logger.debug(
                f"session={session_id} turn={turn}: "
                f"/v1/completions with {len(full_prompt_ids)} prompt tokens"
            )

            resp = await self._client.post(
                f"{self._backend_url}/v1/completions",
                json=completion_params,
                timeout=300,
            )
            resp.raise_for_status()
            completion_resp = resp.json()

            choice = completion_resp["choices"][0]
            response_text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "stop")
            completion_logprobs = choice.get("logprobs")

            response_token_ids: List[int] = choice.get("token_ids") or []
            if not response_token_ids and completion_logprobs:
                response_token_ids = completion_logprobs.get("token_ids", [])
            if not response_token_ids and response_text:
                logger.warning(f"session={session_id} turn={turn}: fallback tokenize response")
                assistant_msg = {"role": "assistant", "content": response_text}
                response_token_ids = await self._tokenizer.tokenize_delta(
                    model, [assistant_msg], tools=tools,
                )

            # Bookkeep — record prompt boundary before appending response
            session.last_prompt_len = len(session.tokens)
            session.append_response(response_token_ids)

            logger.debug(
                f"[TITO] session={session_id} turn={turn}: "
                f"response {len(response_token_ids)} tokens, "
                f"total {len(session.tokens)} tokens, "
                f"has_think={'<think>' in response_text} "
                f"finish={finish_reason}"
            )

            # Parse tool call
            if self._renderer_backend is not None and response_token_ids:
                parsed = self._renderer_backend.parse_response(response_token_ids)
            else:
                parsed = self._tool_parser.parse(response_text)

            # Prefix sanity check
            if self._config.prefix_check:
                await self._run_prefix_check(session_id, turn, session, messages, response_token_ids, tools)

            # Build response
            session.turn += 1
            response = build_chat_completion_response(
                model=model,
                content=parsed["content"],
                tool_calls=parsed["tool_calls"],
                prompt_tokens=len(full_prompt_ids),
                completion_tokens=len(response_token_ids),
                finish_reason=finish_reason,
                logprobs_data=completion_logprobs,
            )
            return JSONResponse(content=response)

        except Exception as e:
            logger.error(f"session={session_id} turn={turn}: ERROR {e}\n{traceback.format_exc()}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

    async def _run_prefix_check(
        self,
        session_id: str,
        turn: int,
        session: SessionState,
        messages: List[Dict[str, Any]],
        response_token_ids: List[int],
        tools: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Run the per-turn tokenization prefix sanity check.

        Validates that the bookkeeping tokens match a fresh tokenization
        of the conversation. Uses the HTTP backend for ground-truth
        comparison (even when the renderer backend is active).
        """
        try:
            if turn == 0:
                check_tokens = await self._tokenizer.tokenize_messages(
                    session.model, messages, add_generation_prompt=False, tools=tools,
                )
                prompt_only = (
                    session.tokens[:-len(response_token_ids)]
                    if response_token_ids
                    else session.tokens
                )
                if check_tokens == list(prompt_only):
                    logger.debug(
                        f"[TITO prefix-check] session={session_id} turn={turn}: "
                        f"OK (prompt {len(prompt_only)} tokens)"
                    )
                else:
                    _check_prefix(session_id, turn, check_tokens, list(prompt_only))
            else:
                prev_seen_idx = session.prev_messages_seen
                obs_start_idx = prev_seen_idx
                while obs_start_idx < len(messages) and messages[obs_start_idx].get("role") == "assistant":
                    obs_start_idx += 1
                obs_msgs_for_check = messages[obs_start_idx:len(messages)]
                if obs_msgs_for_check:
                    oracle_delta = await self._tokenizer.tokenize_delta(
                        session.model, obs_msgs_for_check, tools=tools,
                    )
                    bk_before_obs = len(session.tokens) - len(response_token_ids) - len(oracle_delta)
                    bk_obs = (
                        session.tokens[bk_before_obs:bk_before_obs + len(oracle_delta)]
                        if bk_before_obs >= 0
                        else []
                    )
                    if bk_obs == oracle_delta:
                        logger.debug(
                            f"session={session_id} turn={turn}: "
                            f"OK (obs delta {len(oracle_delta)} tokens)"
                        )
                    else:
                        logger.warning(
                            f"session={session_id} turn={turn}: OBS DELTA MISMATCH"
                        )
                        _check_prefix(session_id, turn, oracle_delta, bk_obs)
        except Exception as e:
            logger.error(
                f"session={session_id} turn={turn}: error {e}\n{traceback.format_exc()}"
            )


def _build_app(
    backend_url: str,
    config: TITOConfig,
    model_name: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI application that implements the TITO proxy.

    Creates endpoints for session-multiplexed chat completions, session
    data retrieval, and session cleanup.

    Args:
        backend_url: Base URL of the vLLM router. Used for ``/tokenize`` and
            ``/v1/completions`` calls.
        config: TITO configuration (parser selection, logging, etc.).
        model_name: Model name for renderer initialization. Required when
            ``config.use_renderer=True``.

    Returns:
        Configured :class:`FastAPI` application.
    """
    app = FastAPI(title="TITO Proxy")
    app.state.sessions: Dict[str, SessionState] = {}

    # Configure file-based logging when log_path is set
    _file_handler = None
    if config.log_path:
        from pathlib import Path

        log_dir = Path(config.log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        _file_handler = logging.FileHandler(log_dir / "tito_proxy.log")
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(_file_handler)

    # Shared HTTP client for connection pooling across all requests
    http_client = httpx.AsyncClient()

    # Select tokenizer backend
    if config.use_renderer:
        tokenizer_backend: TokenizerBackend = RendererTokenizerBackend.create(
            model_name=model_name or "default",
            renderer_name=config.renderer_name,
            backend_url=backend_url,
            client=http_client,
        )
    else:
        tokenizer_backend = HttpTokenizerBackend(backend_url, http_client)

    handler = TITOHandler(
        backend_url=backend_url,
        config=config,
        tokenizer_backend=tokenizer_backend,
        sessions=app.state.sessions,
        http_client=http_client,
    )

    # -----------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/session/{session_id}/v1/models")
    async def models_endpoint(session_id: str, request: Request):
        resp = await http_client.get(f"{backend_url}/v1/models", timeout=10)
        try:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except json.JSONDecodeError:
            return JSONResponse(content={"error": "backend non-JSON"}, status_code=502)

    @app.post("/session/{session_id}/v1/chat/completions")
    async def chat_completions(session_id: str, request: Request):
        """TITO chat completions handler."""
        body = await request.body()
        try:
            req_json = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(content={"error": "invalid JSON"}, status_code=400)
        return await handler.handle_chat_completion(session_id, req_json)

    @app.get("/session/{session_id}/data")
    async def get_session_data(session_id: str):
        """Return session tokens + loss_mask for training."""
        session = app.state.sessions.get(session_id)
        if session is None:
            return JSONResponse(
                content={"error": f"session {session_id} not found"}, status_code=404
            )
        return JSONResponse(content=session.to_dict())

    @app.delete("/session/{session_id}")
    async def delete_session(session_id: str):
        app.state.sessions.pop(session_id, None)
        return {"status": "ok"}

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up the shared HTTP client and log handler on server shutdown."""
        await http_client.aclose()
        if _file_handler is not None:
            logger.removeHandler(_file_handler)
            _file_handler.close()

    return app


class TITOProxyActor:
    """Token-In-Token-Out HTTP proxy server.

    Runs a FastAPI/uvicorn server in a daemon thread that intercepts
    ``/v1/chat/completions`` requests, converts them to token-level
    ``/v1/completions`` calls, and bookkeeps exact ``(token_ids,
    loss_mask)`` per session.

    Can be used in two modes:

    **Standalone** (thread in the current process)::

        proxy = TITOProxyActor(backend_url="http://...:49999", config=cfg)
        url = proxy.start()    # blocks until /health returns 200
        proxy.shutdown()       # stops the server thread

    **Ray actor** (for resource isolation / observability)::

        Actor = ray.remote(TITOProxyActor)
        ref = Actor.remote(backend_url="http://...:49999", config=cfg)
        url = ray.get(ref.start.remote())
        ray.get(ref.shutdown.remote())

    Args:
        backend_url: Base URL of the vLLM router to proxy to.
        config: :class:`TITOConfig` with parser, logging, and port
            settings. Defaults to ``TITOConfig()``.
        model_name: HuggingFace model name for renderer initialization.
            Required when ``config.use_renderer=True``.
    """

    def __init__(
        self,
        backend_url: str,
        config: Optional[TITOConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._backend_url = backend_url
        self._config = config or TITOConfig()
        self._model_name = model_name
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None

        if self._config.port == 0:
            from skyrl.backends.skyrl_train.inference_servers.common import get_open_port

            self._port = get_open_port()
        else:
            self._port = self._config.port

        # For now only allow non-external proxy
        self._host = "127.0.0.1"

    @property
    def url(self) -> str:
        """Base URL of the proxy"""
        return f"http://{self._host}:{self._port}"

    def session_url(self, session_id: str) -> str:
        """Construct a per-session base URL for use as ``OPENAI_BASE_URL``.

        Args:
            session_id: Unique rollout identifier.

        Returns:
            URL like ``http://127.0.0.1:42245/session/{session_id}/v1``.
        """
        return f"{self.url}/session/{session_id}/v1"

    def start(self) -> str:
        """Start the proxy server and block until healthy.

        Spawns a daemon thread running uvicorn, then polls ``GET /health``
        until it returns 200 (up to 30s timeout).

        Returns:
            The proxy base URL (same as :attr:`url`).

        Raises:
            RuntimeError: If the server thread dies or the health check
                times out.
        """
        app = _build_app(self._backend_url, self._config, model_name=self._model_name)
        uvi_config = uvicorn.Config(
            app, host=self._host, port=self._port, log_level="warning"
        )
        self._server = uvicorn.Server(uvi_config)
        self._thread = threading.Thread(
            target=self._server.run, daemon=True, name="tito-proxy"
        )
        self._thread.start()
        self._wait_until_healthy()
        logger.info(f"TITOProxy started at {self.url} → {self._backend_url}")
        return self.url

    def _wait_until_healthy(self, timeout: float = 30.0) -> None:
        """Poll ``GET /health`` until the server responds.

        Args:
            timeout: Maximum wait time in seconds.

        Raises:
            RuntimeError: If the thread dies or timeout is reached.
        """
        health_url = f"{self.url}/health"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._thread and not self._thread.is_alive():
                raise RuntimeError("TITOProxy thread died during startup")
            try:
                with httpx.Client() as client:
                    if client.get(health_url, timeout=1).status_code == 200:
                        return
            except httpx.RequestError:
                time.sleep(0.1)
        raise RuntimeError(f"TITOProxy failed to start within {timeout}s")

    def shutdown(self) -> None:
        """Gracefully stop the proxy server.

        Signals uvicorn to exit and waits up to 5s for the thread to
        join. Safe to call multiple times.
        """
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
