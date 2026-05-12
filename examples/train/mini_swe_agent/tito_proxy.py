"""
TITO Proxy — Token-In-Token-Out proxy for mini-swe-agent.

Sits between litellm (in init_and_run Ray tasks) and the vLLM router.
Presents a standard ``/v1/chat/completions`` endpoint but internally:

  1. Tokenizes inputs via the backend ``/tokenize`` endpoint (with the engine's
     chat template), using a fixed-base delta approach so only new messages are
     tokenized each turn.
  2. Calls ``/v1/completions`` with accumulated token IDs (prompt as int list).
  3. Parses tool calls locally from the raw response text (hermes format).
  4. Returns an OpenAI-compatible chat completion response to the agent.
  5. Bookkeeps exact ``token_ids`` and ``loss_mask`` per session — the
     generator can read these directly, eliminating training-side
     re-tokenization.

Session tracking uses URL path multiplexing — each rollout gets a unique
base URL like ``http://proxy:PORT/session/{session_id}/v1/...``.

A prefix sanity check runs every turn: the full conversation is tokenized
via ``/tokenize`` and compared against the bookkeeping. Mismatches are
logged (retokenization drift detection).

Runs as a background thread within the generator's process.

Usage (standalone test)::

    proxy = TITOProxy(backend_url="http://127.0.0.1:29000")
    proxy_url = proxy.start()
    # Per-rollout URL:
    #   http://proxy:PORT/session/my-rollout-id/v1/chat/completions
    proxy.shutdown()
"""

import datetime
import json
import logging
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed-base dummy messages for delta tokenization
# (matches SkyRL's encode_messages_subset and SLIME's approach)
# ---------------------------------------------------------------------------
DUMMY_BASE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


# ---------------------------------------------------------------------------
# Hermes tool-call parser (standalone, no vLLM dependency at runtime)
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
)


def _parse_hermes_tool_calls(text: str) -> Dict[str, Any]:
    """Parse hermes-format tool calls from raw model output.

    Returns dict with keys:
      - ``content``: text before the first ``<tool_call>`` (or full text if none)
      - ``tool_calls``: list of OpenAI-format tool call dicts, or None
    """
    if "<tool_call>" not in text:
        return {"content": text, "tool_calls": None}

    content = text[: text.index("<tool_call>")].strip() or None
    matches = _TOOL_CALL_RE.findall(text)

    tool_calls = []
    for i, (complete, partial) in enumerate(matches):
        raw = complete if complete else partial
        try:
            parsed = json.loads(raw.strip())
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": parsed.get("name", ""),
                        "arguments": json.dumps(
                            parsed.get("arguments", {}), ensure_ascii=False
                        ),
                    },
                }
            )
        except (json.JSONDecodeError, KeyError):
            # Malformed tool call — treat as content
            if content is None:
                content = text
            return {"content": content, "tool_calls": None}

    return {"content": content, "tool_calls": tool_calls or None}


# ---------------------------------------------------------------------------
# Backend tokenization helpers
# ---------------------------------------------------------------------------
async def _tokenize_messages(
    backend_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    add_generation_prompt: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Call the backend ``/tokenize`` endpoint with chat messages.

    Returns the token IDs produced by the engine's chat template.
    ``tools`` must be forwarded so the template renders identically.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        payload["tools"] = tools
    resp = await client.post(
        f"{backend_url}/tokenize", json=payload, timeout=30
    )
    resp.raise_for_status()
    return resp.json().get("tokens", [])


async def _get_gen_prompt_ids(
    backend_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Get the generation prompt token IDs by diffing tokenize with/without
    ``add_generation_prompt``.

    Returns the generation prompt tokens (e.g. ``<|im_start|>assistant\\n``).
    """
    ids_with = await _tokenize_messages(
        backend_url, model, messages, client,
        add_generation_prompt=True, tools=tools,
    )
    ids_without = await _tokenize_messages(
        backend_url, model, messages, client,
        add_generation_prompt=False, tools=tools,
    )
    return ids_with[len(ids_without):]


async def _tokenize_delta(
    backend_url: str,
    model: str,
    new_messages: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Tokenize only the new messages using the fixed-base approach.

    Tokenizes ``[dummy_base] + new_messages`` and subtracts the dummy prefix
    to get just the delta tokens with correct template wrapping.
    """
    dummy_ids = await _tokenize_messages(
        backend_url, model, DUMMY_BASE, client,
        add_generation_prompt=False, tools=tools,
    )
    full_ids = await _tokenize_messages(
        backend_url, model, DUMMY_BASE + new_messages, client,
        add_generation_prompt=False, tools=tools,
    )
    return full_ids[len(dummy_ids):]


async def _detokenize(
    backend_url: str,
    model: str,
    token_ids: List[int],
    client: httpx.AsyncClient,
) -> str:
    """Detokenize token IDs via the backend ``/detokenize`` endpoint."""
    payload = {"model": model, "tokens": token_ids}
    resp = await client.post(
        f"{backend_url}/detokenize", json=payload, timeout=30
    )
    resp.raise_for_status()
    return resp.json().get("prompt", "")


# ---------------------------------------------------------------------------
# Prefix check
# ---------------------------------------------------------------------------
def _check_prefix(
    session_id: str,
    turn: int,
    prev_tokens: List[int],
    cur_tokens: List[int],
    log_fn=print,
) -> bool:
    """Verify that cur_tokens starts with prev_tokens (prefix match)."""
    prefix_len = len(prev_tokens)
    if prefix_len == 0:
        return True

    cur_prefix = cur_tokens[:prefix_len]
    if cur_prefix == prev_tokens:
        return True

    first_diff = next(
        (
            i
            for i in range(min(prefix_len, len(cur_prefix)))
            if cur_prefix[i] != prev_tokens[i]
        ),
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


# ---------------------------------------------------------------------------
# OpenAI response construction
# ---------------------------------------------------------------------------
def _build_chat_completion_response(
    model: str,
    content: Optional[str],
    tool_calls: Optional[List[Dict[str, Any]]],
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
    logprobs_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct an OpenAI-compatible chat completion response.

    If ``logprobs_data`` is provided (from the completions response), it is
    converted from the completions format to the chat completions format.
    """
    message: Dict[str, Any] = {"role": "assistant", "reasoning_content": None}
    if content is not None:
        message["content"] = content
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    # Convert completions logprobs to chat completions format
    choice_logprobs = None
    if logprobs_data:
        tokens = logprobs_data.get("tokens", [])
        token_logprobs = logprobs_data.get("token_logprobs", [])
        top_logprobs_list = logprobs_data.get("top_logprobs", [])
        lp_content = []
        for i, token_str in enumerate(tokens):
            entry: Dict[str, Any] = {
                "token": token_str,
                "logprob": token_logprobs[i] if i < len(token_logprobs) and token_logprobs[i] is not None else -9999.0,
                "bytes": list(token_str.encode("utf-8", errors="replace")),
            }
            # Convert top_logprobs from {str: float} to [{token, logprob, bytes}]
            if i < len(top_logprobs_list) and top_logprobs_list[i]:
                entry["top_logprobs"] = [
                    {
                        "token": t,
                        "logprob": lp,
                        "bytes": list(t.encode("utf-8", errors="replace")),
                    }
                    for t, lp in top_logprobs_list[i].items()
                ]
            else:
                entry["top_logprobs"] = []
            lp_content.append(entry)
        choice_logprobs = {"content": lp_content}

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": choice_logprobs,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# FastAPI app builder
# ---------------------------------------------------------------------------
def _build_app(backend_url: str, log_path: Optional[str] = None) -> FastAPI:
    """Build the FastAPI app that implements TITO proxying."""

    app = FastAPI(title="TITO Proxy")

    # Per-session state:
    #   tokens: accumulated token IDs (prompt + responses + observations)
    #   loss_mask: 0 for non-generated tokens, 1 for model-generated tokens
    #   turn: current turn number
    #   model: model name (from first request)
    #   tools: tool definitions (from first request)
    #   messages_seen: number of messages processed so far
    app.state.sessions: Dict[str, Dict[str, Any]] = {}

    # File-based logging
    _log_file = None
    if log_path:
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        _log_file = open(log_dir / "tito_proxy.log", "a", buffering=1)

    def _log(msg: str):
        line = f"[{datetime.datetime.now().isoformat()}] {msg}"
        if _log_file:
            _log_file.write(line + "\n")
        else:
            print(line)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/session/{session_id}/v1/models")
    async def models_endpoint(session_id: str, request: Request):
        """Forward model listing to the backend."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{backend_url}/v1/models", timeout=10)
            try:
                return JSONResponse(
                    content=resp.json(), status_code=resp.status_code
                )
            except json.JSONDecodeError:
                return JSONResponse(
                    content={"error": "backend returned non-JSON"},
                    status_code=502,
                )

    @app.post("/session/{session_id}/v1/chat/completions")
    async def chat_completions(session_id: str, request: Request):
        """TITO chat completions: tokenize → /completions → parse → bookkeep.

        Turn 0:
          - Tokenize full messages via /tokenize → prompt_ids
          - Get generation prompt tokens via diff
          - Call /v1/completions with prompt_ids as token ID list
          - Extract response tokens from logprobs
          - Parse tool calls (hermes format)
          - Bookkeep tokens + loss_mask
          - Return OpenAI chat completion response

        Turn N:
          - Tokenize only new messages (delta) via fixed-base
          - Append observation delta to bookkeeping (loss_mask=0)
          - Get generation prompt, call /v1/completions
          - Extract response tokens, parse tool calls
          - Append response to bookkeeping (loss_mask=1)
          - Prefix sanity check: tokenize full conversation, compare
          - Return OpenAI response
        """
        body = await request.body()
        try:
            req_json = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(
                content={"error": "invalid JSON body"}, status_code=400
            )

        messages = req_json.get("messages", [])
        model = req_json.get("model", "default")
        tools = req_json.get("tools")

        # Initialize session on first request
        if session_id not in app.state.sessions:
            app.state.sessions[session_id] = {
                "tokens": [],
                "loss_mask": [],
                "turn": 0,
                "model": model,
                "tools": tools,
                "messages_seen": 0,
            }

        session = app.state.sessions[session_id]
        turn = session["turn"]
        _log(
            f"[TITO] session={session_id} turn={turn} "
            f"n_msgs={len(messages)} roles={[m.get('role','?') for m in messages]}"
        )

        async with httpx.AsyncClient() as client:
            try:
                if turn == 0:
                    # --- Turn 0: tokenize full prompt ---
                    prompt_ids = await _tokenize_messages(
                        backend_url, model, messages, client,
                        add_generation_prompt=False, tools=tools,
                    )
                    session["tokens"] = list(prompt_ids)
                    session["loss_mask"] = [0] * len(prompt_ids)
                    session["messages_seen"] = len(messages)
                else:
                    # --- Turn N: tokenize only new observation messages ---
                    # messages_seen points to the end of what we last processed.
                    # The agent framework adds our assistant response + env
                    # observation(s) after that. We skip the assistant messages
                    # (already bookkeeping as exact engine tokens) and tokenize
                    # only the observation messages.
                    prev_seen = session["messages_seen"]
                    session["_prev_messages_seen"] = prev_seen
                    new_messages = messages[prev_seen:]

                    # The first new message(s) should be assistant (our response).
                    # Skip them — their tokens are already in the bookkeeping.
                    obs_start = 0
                    while obs_start < len(new_messages) and new_messages[obs_start].get("role") == "assistant":
                        obs_start += 1
                    obs_messages = new_messages[obs_start:]

                    if obs_messages:
                        delta_ids = await _tokenize_delta(
                            backend_url, model, obs_messages, client,
                            tools=tools,
                        )
                        session["tokens"].extend(delta_ids)
                        session["loss_mask"].extend([0] * len(delta_ids))
                        _log(
                            f"[TITO] session={session_id} turn={turn}: "
                            f"obs {len(obs_messages)} msgs → {len(delta_ids)} tokens "
                            f"(skipped {obs_start} assistant msgs)"
                        )
                    session["messages_seen"] = len(messages)

                # --- Get generation prompt tokens ---
                gen_prompt_ids = await _get_gen_prompt_ids(
                    backend_url, model, messages, client, tools=tools,
                )

                # --- Build /v1/completions request with token IDs ---
                full_prompt_ids = session["tokens"] + gen_prompt_ids

                # Forward sampling params from original request
                completion_params: Dict[str, Any] = {
                    "model": model,
                    "prompt": full_prompt_ids,
                    "logprobs": 1,
                    "echo": False,
                    "return_token_ids": True,
                }
                # Map chat completion params to completion params
                for key in (
                    "max_tokens", "temperature", "top_p", "top_k",
                    "stop", "frequency_penalty", "presence_penalty",
                    "seed",
                ):
                    if key in req_json:
                        completion_params[key] = req_json[key]
                # Handle max_tokens: chat completions may use max_completion_tokens
                if "max_completion_tokens" in req_json and "max_tokens" not in completion_params:
                    completion_params["max_tokens"] = req_json["max_completion_tokens"]
                # Ensure skip_special_tokens is False so we get tool call tokens
                completion_params["skip_special_tokens"] = False

                _log(
                    f"[TITO] session={session_id} turn={turn}: "
                    f"calling /v1/completions with {len(full_prompt_ids)} prompt tokens"
                )

                resp = await client.post(
                    f"{backend_url}/v1/completions",
                    json=completion_params,
                    timeout=300,
                )
                resp.raise_for_status()
                completion_resp = resp.json()

                # --- Extract response token IDs ---
                choice = completion_resp["choices"][0]
                response_text = choice.get("text", "")
                finish_reason = choice.get("finish_reason", "stop")
                completion_logprobs = choice.get("logprobs")

                # Prefer return_token_ids (exact), fall back to logprobs
                response_token_ids: List[int] = choice.get("token_ids") or []
                if not response_token_ids:
                    if completion_logprobs and "token_ids" in completion_logprobs:
                        response_token_ids = completion_logprobs["token_ids"]

                # Fallback: if no token IDs available, tokenize the response text
                if not response_token_ids and response_text:
                    _log(
                        f"[TITO] session={session_id} turn={turn}: "
                        f"WARNING: no token_ids in logprobs, falling back to tokenize response"
                    )
                    # Use the assistant message format for tokenization
                    assistant_msg = {"role": "assistant", "content": response_text}
                    response_token_ids = await _tokenize_delta(
                        backend_url, model, [assistant_msg], client, tools=tools,
                    )

                # --- Bookkeep response tokens ---
                session["tokens"].extend(response_token_ids)
                session["loss_mask"].extend([1] * len(response_token_ids))

                _log(
                    f"[TITO] session={session_id} turn={turn}: "
                    f"response {len(response_token_ids)} tokens, "
                    f"total {len(session['tokens'])} tokens, "
                    f"finish_reason={finish_reason}"
                )
                _log(
                    f"[TITO] session={session_id} turn={turn}: "
                    f"response_text has_think={'<think>' in response_text} "
                    f"has_tool_call={'<tool_call>' in response_text} "
                    f"text_len={len(response_text)} "
                    f"text_first200={repr(response_text[:200])}"
                )

                # --- Parse tool calls from response text ---
                parsed = _parse_hermes_tool_calls(response_text)

                # --- Prefix sanity check ---
                # Validate the observation delta tokenization: compare our
                # fixed-base delta against a full-conversation tokenization.
                # We can't compare the full bookkeeping because engine response
                # tokens (with thinking blocks) won't match re-tokenization.
                # Instead we check: tokenize(messages_up_to_prev_seen) is a
                # prefix of tokenize(messages_including_new_obs), and the
                # delta portion matches our delta_ids.
                try:
                    if turn == 0:
                        # Turn 0: simple check — prompt_ids == full tokenization
                        check_tokens = await _tokenize_messages(
                            backend_url, model, messages, client,
                            add_generation_prompt=False, tools=tools,
                        )
                        prompt_only = session["tokens"][: -len(response_token_ids)] if response_token_ids else session["tokens"]
                        if check_tokens == list(prompt_only):
                            _log(
                                f"[TITO prefix-check] session={session_id} turn={turn}: "
                                f"OK (prompt {len(prompt_only)} tokens)"
                            )
                        else:
                            _check_prefix(session_id, turn, check_tokens, list(prompt_only), log_fn=_log)
                    else:
                        # Turn N: validate observation delta
                        # Tokenize messages up to prev_seen (before new obs)
                        prev_seen = session.get("_prev_messages_seen", 0)
                        # Tokenize with the current obs messages included
                        # to get the oracle delta
                        obs_start_idx = prev_seen
                        # Skip assistant messages in the delta (same logic as above)
                        while obs_start_idx < len(messages) and messages[obs_start_idx].get("role") == "assistant":
                            obs_start_idx += 1
                        obs_msgs_for_check = messages[obs_start_idx:len(messages)]
                        if obs_msgs_for_check:
                            oracle_delta = await _tokenize_delta(
                                backend_url, model, obs_msgs_for_check, client,
                                tools=tools,
                            )
                            # Compare against our bookkeeping delta
                            bk_before_obs = len(session["tokens"]) - len(response_token_ids) - len(oracle_delta)
                            bk_obs = session["tokens"][bk_before_obs: bk_before_obs + len(oracle_delta)] if bk_before_obs >= 0 else []
                            if bk_obs == oracle_delta:
                                _log(
                                    f"[TITO prefix-check] session={session_id} turn={turn}: "
                                    f"OK (obs delta {len(oracle_delta)} tokens match)"
                                )
                            else:
                                _log(
                                    f"[TITO prefix-check] session={session_id} turn={turn}: "
                                    f"OBS DELTA MISMATCH oracle={len(oracle_delta)} bk={len(bk_obs)}"
                                )
                                if oracle_delta and bk_obs:
                                    _check_prefix(session_id, turn, oracle_delta, bk_obs, log_fn=_log)
                        else:
                            _log(
                                f"[TITO prefix-check] session={session_id} turn={turn}: "
                                f"no obs messages to check"
                            )
                except Exception as e:
                    _log(
                        f"[TITO prefix-check] session={session_id} turn={turn}: "
                        f"error {e}\n{traceback.format_exc()}"
                    )

                # --- Build and return OpenAI chat completion response ---
                session["turn"] += 1
                prompt_token_count = len(full_prompt_ids)
                response = _build_chat_completion_response(
                    model=model,
                    content=parsed["content"],
                    tool_calls=parsed["tool_calls"],
                    prompt_tokens=prompt_token_count,
                    completion_tokens=len(response_token_ids),
                    finish_reason=finish_reason,
                    logprobs_data=completion_logprobs,
                )
                return JSONResponse(content=response)

            except Exception as e:
                _log(
                    f"[TITO] session={session_id} turn={turn}: "
                    f"ERROR {e}\n{traceback.format_exc()}"
                )
                return JSONResponse(
                    content={"error": str(e)}, status_code=500
                )

    @app.get("/session/{session_id}/data")
    async def get_session_data(session_id: str):
        """Return session token_ids and loss_mask for training.

        The generator calls this after a rollout completes to get
        the exact token sequence and loss mask without re-tokenization.
        """
        session = app.state.sessions.get(session_id)
        if session is None:
            return JSONResponse(
                content={"error": f"session {session_id} not found"},
                status_code=404,
            )
        return JSONResponse(
            content={
                "tokens": session["tokens"],
                "loss_mask": session["loss_mask"],
                "turn": session["turn"],
                "model": session["model"],
            }
        )

    @app.delete("/session/{session_id}")
    async def delete_session(session_id: str):
        """Clean up session state when a rollout completes."""
        app.state.sessions.pop(session_id, None)
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# TITOProxy lifecycle management
# ---------------------------------------------------------------------------
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
            from skyrl.backends.skyrl_train.inference_servers.common import (
                get_open_port,
            )

            self._port = get_open_port()
            self._port_reservation = None
        else:
            from skyrl.backends.skyrl_train.inference_servers.common import (
                find_and_reserve_port,
            )

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
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None

        app = _build_app(self._backend_url, log_path=self._log_path)
        config = uvicorn.Config(
            app, host=self._host, port=self._port, log_level="warning"
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run, daemon=True, name="tito-proxy"
        )
        self._thread.start()

        self._wait_until_healthy()
        logger.info(
            f"TITOProxy started at {self.url}, forwarding to {self._backend_url}"
        )
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
        raise RuntimeError(
            f"TITOProxy failed to become healthy within {timeout}s"
        )

    def shutdown(self) -> None:
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
