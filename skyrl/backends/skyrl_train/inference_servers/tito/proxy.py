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
      │  POST {backend}/tokenize        ← tokenize observations
      │  POST {backend}/v1/completions  ← generate with token IDs
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

import datetime
import json
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import TITOConfig
from .response import build_chat_completion_response
from .session import SessionState
from .tokenizer import get_gen_prompt_ids, tokenize_delta, tokenize_messages
from .tool_parsers import get_tool_parser

logger = logging.getLogger(__name__)


def _check_prefix(
    session_id: str,
    turn: int,
    prev_tokens: List[int],
    cur_tokens: List[int],
    log_fn=print,
) -> bool:
    """Verify that *cur_tokens* starts with *prev_tokens* (prefix match).

    Used by the per-turn sanity check to detect tokenization drift.
    Logs a detailed mismatch report (with a 10-token window around the
    first divergence) via *log_fn* when the prefix doesn't match.

    Args:
        session_id: Session identifier (for log messages).
        turn: Current turn number (for log messages).
        prev_tokens: Expected prefix token IDs.
        cur_tokens: Actual token IDs to check against.
        log_fn: Callable for diagnostic output (default: ``print``).

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
    log_fn(
        f"[TITO prefix-check] session={session_id} turn={turn}: "
        f"MISMATCH at token {first_diff}/{prefix_len} (cur total={len(cur_tokens)})\n"
        f"  prev[{start}:{end}] = {prev_tokens[start:end]}\n"
        f"  cur [{start}:{end}] = {cur_tokens[start:end]}"
    )
    return False


def _build_app(
    backend_url: str,
    config: TITOConfig,
) -> FastAPI:
    """Build the FastAPI application that implements the TITO proxy.

    Creates endpoints for session-multiplexed chat completions, session
    data retrieval, and session cleanup. All state is held in
    ``app.state.sessions`` (a dict of :class:`SessionState` keyed by
    session ID).

    Args:
        backend_url: Base URL of the vLLM router (e.g.
            ``http://10.166.15.194:49999``). Used for ``/tokenize`` and
            ``/v1/completions`` calls.
        config: TITO configuration (parser selection, logging, etc.).

    Returns:
        Configured :class:`FastAPI` application.
    """

    app = FastAPI(title="TITO Proxy")
    app.state.sessions: Dict[str, SessionState] = {}

    tool_parser = get_tool_parser(config.tool_call_parser)

    # File-based logging
    _log_file = None
    if config.log_path:
        log_dir = Path(config.log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        _log_file = open(log_dir / "tito_proxy.log", "a", buffering=1)

    def _log(msg: str):
        line = f"[{datetime.datetime.now().isoformat()}] {msg}"
        if _log_file:
            _log_file.write(line + "\n")
        else:
            print(line)

    # -----------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/session/{session_id}/v1/models")
    async def models_endpoint(session_id: str, request: Request):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{backend_url}/v1/models", timeout=10)
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

        messages = req_json.get("messages", [])
        model = req_json.get("model", "default")
        tools = req_json.get("tools")

        if session_id not in app.state.sessions:
            app.state.sessions[session_id] = SessionState(model=model, tools=tools)

        session = app.state.sessions[session_id]
        turn = session.turn
        _log(
            f"[TITO] session={session_id} turn={turn} "
            f"n_msgs={len(messages)} roles={[m.get('role','?') for m in messages]}"
        )

        async with httpx.AsyncClient() as client:
            try:
                # --- Tokenize input ---
                if turn == 0:
                    prompt_ids = await tokenize_messages(
                        backend_url, model, messages, client,
                        add_generation_prompt=False, tools=tools,
                    )
                    session.tokens = list(prompt_ids)
                    session.loss_mask = [0] * len(prompt_ids)
                    session.messages_seen = len(messages)
                else:
                    prev_seen = session.messages_seen
                    session._prev_messages_seen = prev_seen
                    new_messages = messages[prev_seen:]
                    obs_start = 0
                    while obs_start < len(new_messages) and new_messages[obs_start].get("role") == "assistant":
                        obs_start += 1
                    obs_messages = new_messages[obs_start:]
                    if obs_messages:
                        delta_ids = await tokenize_delta(
                            backend_url, model, obs_messages, client, tools=tools,
                        )
                        session.tokens.extend(delta_ids)
                        session.loss_mask.extend([0] * len(delta_ids))
                        _log(
                            f"[TITO] session={session_id} turn={turn}: "
                            f"obs {len(obs_messages)} msgs → {len(delta_ids)} tokens "
                            f"(skipped {obs_start} assistant msgs)"
                        )
                    session.messages_seen = len(messages)

                # --- Generation prompt ---
                gen_prompt_ids = await get_gen_prompt_ids(
                    backend_url, model, messages, client, tools=tools,
                )
                full_prompt_ids = session.tokens + gen_prompt_ids

                # --- /v1/completions with token IDs ---
                completion_params: Dict[str, Any] = {
                    "model": model,
                    "prompt": full_prompt_ids,
                    "logprobs": 1,
                    "echo": False,
                    "return_token_ids": True,
                    "skip_special_tokens": False,
                }
                for key in ("max_tokens", "temperature", "top_p", "top_k",
                            "stop", "frequency_penalty", "presence_penalty", "seed"):
                    if key in req_json:
                        completion_params[key] = req_json[key]
                if "max_completion_tokens" in req_json and "max_tokens" not in completion_params:
                    completion_params["max_tokens"] = req_json["max_completion_tokens"]

                _log(
                    f"[TITO] session={session_id} turn={turn}: "
                    f"/v1/completions with {len(full_prompt_ids)} prompt tokens"
                )

                resp = await client.post(
                    f"{backend_url}/v1/completions",
                    json=completion_params,
                    timeout=300,
                )
                resp.raise_for_status()
                completion_resp = resp.json()

                # --- Extract response ---
                choice = completion_resp["choices"][0]
                response_text = choice.get("text", "")
                finish_reason = choice.get("finish_reason", "stop")
                completion_logprobs = choice.get("logprobs")

                response_token_ids: List[int] = choice.get("token_ids") or []
                if not response_token_ids and completion_logprobs:
                    response_token_ids = completion_logprobs.get("token_ids", [])
                if not response_token_ids and response_text:
                    _log(f"[TITO] session={session_id} turn={turn}: WARNING fallback tokenize response")
                    assistant_msg = {"role": "assistant", "content": response_text}
                    response_token_ids = await tokenize_delta(
                        backend_url, model, [assistant_msg], client, tools=tools,
                    )

                # --- Bookkeep ---
                session.tokens.extend(response_token_ids)
                session.loss_mask.extend([1] * len(response_token_ids))

                _log(
                    f"[TITO] session={session_id} turn={turn}: "
                    f"response {len(response_token_ids)} tokens, "
                    f"total {len(session.tokens)} tokens, "
                    f"has_think={'<think>' in response_text} "
                    f"finish={finish_reason}"
                )

                # --- Parse tool calls ---
                parsed = tool_parser.parse(response_text)

                # --- Prefix sanity check ---
                if config.prefix_check:
                    try:
                        if turn == 0:
                            check_tokens = await tokenize_messages(
                                backend_url, model, messages, client,
                                add_generation_prompt=False, tools=tools,
                            )
                            prompt_only = session.tokens[:-len(response_token_ids)] if response_token_ids else session.tokens
                            if check_tokens == list(prompt_only):
                                _log(f"[TITO prefix-check] session={session_id} turn={turn}: OK (prompt {len(prompt_only)} tokens)")
                            else:
                                _check_prefix(session_id, turn, check_tokens, list(prompt_only), log_fn=_log)
                        else:
                            prev_seen_idx = session._prev_messages_seen
                            obs_start_idx = prev_seen_idx
                            while obs_start_idx < len(messages) and messages[obs_start_idx].get("role") == "assistant":
                                obs_start_idx += 1
                            obs_msgs_for_check = messages[obs_start_idx:len(messages)]
                            if obs_msgs_for_check:
                                oracle_delta = await tokenize_delta(
                                    backend_url, model, obs_msgs_for_check, client, tools=tools,
                                )
                                bk_before_obs = len(session.tokens) - len(response_token_ids) - len(oracle_delta)
                                bk_obs = session.tokens[bk_before_obs:bk_before_obs + len(oracle_delta)] if bk_before_obs >= 0 else []
                                if bk_obs == oracle_delta:
                                    _log(f"[TITO prefix-check] session={session_id} turn={turn}: OK (obs delta {len(oracle_delta)} tokens)")
                                else:
                                    _log(f"[TITO prefix-check] session={session_id} turn={turn}: OBS DELTA MISMATCH")
                                    _check_prefix(session_id, turn, oracle_delta, bk_obs, log_fn=_log)
                    except Exception as e:
                        _log(f"[TITO prefix-check] session={session_id} turn={turn}: error {e}\n{traceback.format_exc()}")

                # --- Build response ---
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
                _log(f"[TITO] session={session_id} turn={turn}: ERROR {e}\n{traceback.format_exc()}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

    @app.get("/session/{session_id}/data")
    async def get_session_data(session_id: str):
        """Return session tokens + loss_mask for training."""
        session = app.state.sessions.get(session_id)
        if session is None:
            return JSONResponse(content={"error": f"session {session_id} not found"}, status_code=404)
        return JSONResponse(content=session.to_dict())

    @app.delete("/session/{session_id}")
    async def delete_session(session_id: str):
        app.state.sessions.pop(session_id, None)
        return {"status": "ok"}

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
        backend_url: Base URL of the vLLM router to proxy to (e.g.
            ``http://10.166.15.194:49999``).
        config: :class:`TITOConfig` with parser, logging, and port
            settings. Defaults to ``TITOConfig()`` (disabled by default,
            but the constructor doesn't check ``enabled`` — the caller
            in ``build_new_inference_client`` gates on that).
    """

    def __init__(self, backend_url: str, config: Optional[TITOConfig] = None):
        self._backend_url = backend_url
        self._config = config or TITOConfig()
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None

        if self._config.port == 0:
            from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
            self._port = get_open_port()
        else:
            self._port = self._config.port

        self._host = "127.0.0.1"

    @property
    def url(self) -> str:
        """Base URL of the proxy (e.g. ``http://127.0.0.1:42245``)."""
        return f"http://{self._host}:{self._port}"

    def session_url(self, session_id: str) -> str:
        """Construct a per-session base URL for use as ``OPENAI_BASE_URL``.

        Args:
            session_id: Unique rollout identifier (e.g.
                ``instance_id_rep_id_step``).

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
        app = _build_app(self._backend_url, self._config)
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
