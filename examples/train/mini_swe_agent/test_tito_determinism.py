#!/usr/bin/env python3
"""
Deterministic comparison test: TITO proxy vs direct /chat/completions.

Sends the same prompt through both paths with temperature=0, seed=42,
and compares the response content token-by-token.

Usage:
    # Start the generate script first (to get vLLM engines running), then:
    python examples/train/mini_swe_agent/test_tito_determinism.py \
        --backend-url http://127.0.0.1:8001 \
        --model Qwen/Qwen3-1.7B

    # Or with a custom prompt:
    python examples/train/mini_swe_agent/test_tito_determinism.py \
        --backend-url http://127.0.0.1:8001 \
        --model Qwen/Qwen3-1.7B \
        --max-turns 3
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a helpful assistant that can interact with a computer shell. "
        "Your response must contain exactly ONE bash code block.\n"
        "Include a THOUGHT section before your command.\n"
    ),
}

USER_MSG = {
    "role": "user",
    "content": (
        "Find all Python files in the current directory that contain the word 'import'. "
        "Show a count of how many files match."
    ),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run"},
                },
                "required": ["command"],
            },
        },
    }
]

OBSERVATION = {
    "role": "user",
    "content": "Command output:\n42 files found\n\nProvide your next command or say DONE.",
}

TOOL_OBSERVATION = {
    "role": "tool",
    "content": "42 files found",
}

SAMPLING_PARAMS = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 512,
}


async def call_chat_completions(
    backend_url: str, model: str, messages: list, tools: list | None = None
) -> dict:
    """Call /v1/chat/completions directly on the backend."""
    payload = {
        "model": model,
        "messages": messages,
        "skip_special_tokens": False,
        "logprobs": True,
        "top_logprobs": 1,
        **SAMPLING_PARAMS,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{backend_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


async def call_tito_proxy(
    proxy_url: str, session_id: str, model: str, messages: list, tools: list | None = None
) -> dict:
    """Call /v1/chat/completions via the TITO proxy."""
    payload = {
        "model": model,
        "messages": messages,
        "skip_special_tokens": False,
        "logprobs": True,
        "top_logprobs": 1,
        **SAMPLING_PARAMS,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{proxy_url}/session/{session_id}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


async def get_tito_session_data(proxy_url: str, session_id: str) -> dict:
    """Get session tokens + loss_mask from the TITO proxy."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{proxy_url}/session/{session_id}/data",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()


def extract_content(resp: dict) -> str:
    """Extract assistant content from a chat completion response."""
    return resp["choices"][0]["message"].get("content") or ""


def extract_tool_calls(resp: dict) -> list | None:
    return resp["choices"][0]["message"].get("tool_calls")


def extract_logprob_tokens(resp: dict) -> list[str]:
    """Extract token strings from logprobs."""
    lp = resp["choices"][0].get("logprobs")
    if not lp or not lp.get("content"):
        return []
    return [entry["token"] for entry in lp["content"]]


def compare_responses(label: str, direct: dict, tito: dict):
    """Compare two chat completion responses."""
    d_content = extract_content(direct)
    t_content = extract_content(tito)
    d_tc = extract_tool_calls(direct)
    t_tc = extract_tool_calls(tito)
    d_tokens = extract_logprob_tokens(direct)
    t_tokens = extract_logprob_tokens(tito)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Content comparison
    content_match = d_content == t_content
    print(f"  Content match: {content_match}")
    print(f"  Direct content length: {len(d_content)}")
    print(f"  TITO content length:   {len(t_content)}")
    if not content_match:
        # Find first diff
        for i in range(min(len(d_content), len(t_content))):
            if d_content[i] != t_content[i]:
                print(f"  First diff at char {i}:")
                print(f"    Direct: ...{repr(d_content[max(0,i-20):i+20])}...")
                print(f"    TITO:   ...{repr(t_content[max(0,i-20):i+20])}...")
                break
        else:
            shorter = "direct" if len(d_content) < len(t_content) else "tito"
            print(f"  {shorter} is shorter by {abs(len(d_content) - len(t_content))} chars")

    # Tool calls comparison
    if d_tc or t_tc:
        tc_match = json.dumps(d_tc, sort_keys=True) == json.dumps(t_tc, sort_keys=True)
        print(f"  Tool calls match: {tc_match}")
        if not tc_match:
            print(f"    Direct: {json.dumps(d_tc)[:200]}")
            print(f"    TITO:   {json.dumps(t_tc)[:200]}")

    # Logprob tokens comparison
    if d_tokens or t_tokens:
        tokens_match = d_tokens == t_tokens
        print(f"  Logprob tokens match: {tokens_match} (direct={len(d_tokens)}, tito={len(t_tokens)})")
        if not tokens_match:
            for i in range(min(len(d_tokens), len(t_tokens))):
                if d_tokens[i] != t_tokens[i]:
                    print(f"  First token diff at position {i}:")
                    ctx = 3
                    print(f"    Direct[{max(0,i-ctx)}:{i+ctx}]: {d_tokens[max(0,i-ctx):i+ctx]}")
                    print(f"    TITO[{max(0,i-ctx)}:{i+ctx}]:   {t_tokens[max(0,i-ctx):i+ctx]}")
                    break

    # Usage comparison
    d_usage = direct.get("usage", {})
    t_usage = tito.get("usage", {})
    print(f"  Usage - direct: {d_usage}")
    print(f"  Usage - tito:   {t_usage}")

    return content_match


async def run_test(args):
    backend_url = args.backend_url
    model = args.model
    max_turns = args.max_turns

    # Start TITO proxy
    from examples.train.mini_swe_agent.tito_proxy import TITOProxy

    proxy = TITOProxy(
        backend_url=backend_url,
        log_path=args.log_path,
    )
    proxy.start()
    proxy_url = proxy.url
    print(f"TITO proxy started at {proxy_url}")

    # Verify backend is reachable
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{backend_url}/v1/models", timeout=5)
            models = resp.json()
            print(f"Backend models: {[m['id'] for m in models.get('data', [])]}")
        except Exception as e:
            print(f"ERROR: Backend not reachable at {backend_url}: {e}")
            proxy.shutdown()
            return

    session_id = "test_determinism"
    messages = [SYSTEM_MSG, USER_MSG]
    all_match = True

    for turn in range(max_turns):
        print(f"\n--- Turn {turn} ({len(messages)} messages) ---")

        # Call both paths
        t0 = time.time()
        direct_resp = await call_chat_completions(
            backend_url, model, messages, tools=TOOLS
        )
        t_direct = time.time() - t0

        t0 = time.time()
        tito_resp = await call_tito_proxy(
            proxy_url, session_id, model, messages, tools=TOOLS
        )
        t_tito = time.time() - t0

        print(f"  Direct: {t_direct:.2f}s, TITO: {t_tito:.2f}s")

        match = compare_responses(f"Turn {turn}", direct_resp, tito_resp)
        all_match = all_match and match

        # Build next turn messages
        d_content = extract_content(direct_resp)
        d_tc = extract_tool_calls(direct_resp)

        # Use direct response as ground truth for next turn
        assistant_msg = {"role": "assistant", "content": d_content}
        if d_tc:
            assistant_msg["tool_calls"] = d_tc
        messages.append(assistant_msg)

        # Add observation
        if d_tc:
            messages.append(TOOL_OBSERVATION)
        else:
            messages.append(OBSERVATION)

    # Get TITO session data
    session_data = await get_tito_session_data(proxy_url, session_id)
    print(f"\n--- TITO Session Data ---")
    print(f"  Total tokens: {len(session_data['tokens'])}")
    print(f"  Loss mask sum: {sum(session_data['loss_mask'])} / {len(session_data['loss_mask'])}")
    print(f"  Turns: {session_data['turn']}")

    proxy.shutdown()

    print(f"\n{'='*60}")
    print(f"  ALL TURNS MATCH: {all_match}")
    print(f"{'='*60}")
    return all_match


def main():
    parser = argparse.ArgumentParser(description="Test TITO proxy determinism")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8001",
                        help="vLLM backend URL")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="Model name")
    parser.add_argument("--max-turns", type=int, default=1,
                        help="Number of turns to test")
    parser.add_argument("--log-path", default="/tmp/tito_test",
                        help="Log directory for TITO proxy")
    args = parser.parse_args()

    result = asyncio.run(run_test(args))
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
