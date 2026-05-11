"""
TITO Proxy — a lightweight OpenAI-compatible HTTP proxy for mini-swe-agent.

Sits between litellm (in init_and_run Ray tasks) and the vLLM router,
presenting a standard /v1/chat/completions endpoint. Currently forwards
requests to the backend router as-is; the TITO conversion logic will be
added here later (render → generate with token IDs → detokenize).

Session tracking uses URL path multiplexing — each rollout gets a unique
base URL like http://proxy:PORT/session/{session_id}/v1/... so the proxy
can identify sessions without any agent-side changes.

Prefix sanity check: on every chat/completions request the proxy also calls
the backend's /tokenize endpoint to tokenize the full conversation. It then
verifies that the token IDs share a common prefix with the bookkeeping from
the previous turn. A mismatch indicates retokenization drift (e.g. the chat
template stripping thinking tokens on re-encode).

Runs as a background thread within the generator's process.

Usage (standalone test):
    proxy = TITOProxy(backend_url="http://127.0.0.1:29000")
    proxy_url = proxy.start()
    # Per-rollout URL:
    #   http://proxy:PORT/session/my-rollout-id/v1/chat/completions
    proxy.shutdown()
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)


async def _tokenize_messages(
    backend_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    add_generation_prompt: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Call the backend /tokenize endpoint with chat messages.

    Returns the token IDs produced by the engine's chat template.
    ``tools`` must be forwarded from the original request so the template
    renders the system message identically (Qwen3 embeds tool descriptions
    into the system turn when tools are present).
    """
    payload = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        payload["tools"] = tools
    resp = await client.post(f"{backend_url}/tokenize", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("tokens", [])


def _check_prefix(
    session_id: str,
    turn: int,
    prev_tokens: List[int],
    cur_tokens: List[int],
) -> bool:
    """Verify that cur_tokens starts with prev_tokens (prefix match).

    Returns True if the prefix matches, False otherwise.
    Logs details on mismatch.
    """
    prefix_len = len(prev_tokens)
    if prefix_len == 0:
        return True

    cur_prefix = cur_tokens[:prefix_len]
    if cur_prefix == prev_tokens:
        return True

    # Find first divergence point
    first_diff = next(
        (i for i in range(min(prefix_len, len(cur_prefix))) if cur_prefix[i] != prev_tokens[i]),
        min(prefix_len, len(cur_prefix)),
    )
    logger.warning(
        f"[TITO prefix-check] session={session_id} turn={turn}: "
        f"MISMATCH at token {first_diff}/{prefix_len}. "
        f"prev[{first_diff}:]={prev_tokens[first_diff:first_diff+5]} "
        f"cur[{first_diff}:]={cur_prefix[first_diff:first_diff+5]}"
    )
    return False


def _build_app(backend_url: str) -> FastAPI:
    """Build the FastAPI app that proxies to the backend."""

    app = FastAPI(title="TITO Proxy")

    # Per-session state for future TITO logic (token history, KV prefix, etc.)
    # Key: session_id (from URL path), Value: dict of session state
    app.state.sessions = {}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/session/{session_id}/v1/models")
    async def models(session_id: str, request: Request):
        """Forward model listing to the backend."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{backend_url}/v1/models", timeout=10)
            try:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except json.JSONDecodeError:
                return JSONResponse(content={"error": "backend returned non-JSON"}, status_code=502)

    @app.post("/session/{session_id}/v1/chat/completions")
    async def chat_completions(session_id: str, request: Request):
        """Forward chat completions to the backend router.

        On each turn the proxy also tokenizes the full conversation via the
        backend's /tokenize endpoint and checks that the result shares a
        common prefix with the bookkeeping from the previous turn. A mismatch
        is logged as a warning (retokenization drift).
        """
        if session_id not in app.state.sessions:
            app.state.sessions[session_id] = {"turn": 0, "prefix_tokens": []}

        session = app.state.sessions[session_id]
        turn = session["turn"]
        print(f"Received request for session {session_id}, turn {turn}")

        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        try:
            req_json = json.loads(body)
            is_stream = req_json.get("stream", False)
        except (json.JSONDecodeError, UnicodeDecodeError):
            req_json = None
            is_stream = False

        # --- Prefix sanity check (non-blocking: failures are logged, not raised) ---
        # Tokenize the full conversation WITHOUT add_generation_prompt so we get
        # just the history tokens. On the next turn the conversation will have
        # more messages appended; its prefix (all previous messages) should
        # tokenize identically if the chat template is stable.
        if req_json and "messages" in req_json:
            try:
                async with httpx.AsyncClient() as tok_client:
                    model = req_json.get("model", "default")
                    cur_tokens = await _tokenize_messages(
                        backend_url, model, req_json["messages"], tok_client,
                        add_generation_prompt=False,
                        tools=req_json.get("tools"),
                    )
                prev_tokens = session["prefix_tokens"]
                match = _check_prefix(session_id, turn, prev_tokens, cur_tokens)
                if match:
                    logger.info(
                        f"[TITO prefix-check] session={session_id} turn={turn}: "
                        f"OK (prefix {len(prev_tokens)} tokens, total {len(cur_tokens)})"
                    )
                # Bookkeep the current full tokenization as the expected prefix
                # for the next turn's conversation.
                session["prefix_tokens"] = cur_tokens
                print(len(session["prefix_tokens"]))
            except Exception as e:
                logger.warning(f"[TITO prefix-check] session={session_id} turn={turn}: error {e}")

        # --- Forward to backend ---
        if is_stream:
            async def stream_response():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{backend_url}/v1/chat/completions",
                        content=body,
                        headers=headers,
                        timeout=300,
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk

            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{backend_url}/v1/chat/completions",
                    content=body,
                    headers=headers,
                    timeout=300,
                )
                session["turn"] += 1
                try:
                    return JSONResponse(content=resp.json(), status_code=resp.status_code)
                except json.JSONDecodeError:
                    return JSONResponse(content={"error": "backend returned non-JSON"}, status_code=502)

    @app.delete("/session/{session_id}")
    async def delete_session(session_id: str):
        """Clean up session state when a rollout completes."""
        app.state.sessions.pop(session_id, None)
        return {"status": "ok"}

    return app


class TITOProxy:
    """Background-thread proxy server.

    Runs uvicorn in a daemon thread so it works inside Ray workers
    without multiprocessing/pickling issues.
    """

    def __init__(
        self,
        backend_url: str,
        host: str = "127.0.0.1",
        port: int = 0,
        log_path: Optional[str] = None,
    ):
        self._backend_url = backend_url
        self._host = host
        self._log_path = log_path
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None

        if port == 0:
            from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
            self._port = get_open_port()
            self._port_reservation = None
        else:
            from skyrl.backends.skyrl_train.inference_servers.common import find_and_reserve_port
            reserved_port, self._port_reservation = find_and_reserve_port(port)
            self._port = reserved_port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def session_url(self, session_id: str) -> str:
        """Return a per-session base URL for use as OPENAI_BASE_URL."""
        return f"{self.url}/session/{session_id}/v1"

    def start(self) -> str:
        """Start the proxy in a background thread and return the base URL once healthy."""
        # Release port reservation right before uvicorn binds
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None

        app = _build_app(self._backend_url)
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._server.run, daemon=True, name="tito-proxy")
        self._thread.start()

        self._wait_until_healthy()
        logger.info(f"TITOProxy started at {self.url}, forwarding to {self._backend_url}")
        return self.url

    def _wait_until_healthy(self, timeout: float = 30.0) -> None:
        health_url = f"{self.url}/health"
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._thread is not None and not self._thread.is_alive():
                raise RuntimeError("TITOProxy thread died during startup")
            try:
                with httpx.Client() as client:
                    if client.get(health_url, timeout=1).status_code == 200:
                        return
            except httpx.RequestError:
                time.sleep(0.1)
        raise RuntimeError(f"TITOProxy failed to become healthy within {timeout}s")

    def shutdown(self) -> None:
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
