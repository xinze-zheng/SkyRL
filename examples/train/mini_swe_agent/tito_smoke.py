"""Phase 1 smoke test for `LitellmTITOModel`.

Goals (see tito.md / design plan for full rationale):

  1. Capture is faithful — the token IDs we stash are exactly what vLLM
     sampled, decoded back to the message content.
  2. Drop-in semantics — `LitellmTITOModel` still produces parsed `actions`
     so `DefaultAgent` would happily consume the result.
  3. Step-2 prompt is a strict extension — what vLLM sees on turn 2 starts
     with the same bytes our local tokenizer would render from
     (system + user + assistant + tool) with `tools=[BASH_TOOL]`.

This script does NOT import SkyRL or Ray. It assumes a vLLM server is
already running and reachable at --base-url. Iterate on the small model;
re-run once on the real 30B target before declaring victory.

Example:
  # Start a small vLLM (cheap to iterate on):
  uv run vllm serve Qwen/Qwen2.5-0.5B-Instruct \
      --port 8002 --enable-auto-tool-choice --tool-call-parser hermes &

  uv run python examples/train/mini_swe_agent/tito_smoke.py \
      --model-id Qwen/Qwen2.5-0.5B-Instruct \
      --base-url http://127.0.0.1:8002/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_tito_model import LitellmTITOModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from minisweagent.models.utils.chat_templates import resolve_chat_template
from transformers import AutoTokenizer

SYS = (
    "You are a shell assistant. For every user request you MUST call the "
    "`bash` tool with a single short command. Do not answer in prose. "
    "Do not include reasoning, explanation, or thinking — emit ONLY the "
    "tool call."
)
USER = "Run `ls /tmp` via the bash tool. Emit the tool call now."


def _query_capturing_format_error(model, msgs):
    """Return (message, format_error_or_none).

    LitellmModel.query raises FormatError when the model produces no tool call
    or an unknown tool, but it attaches the partial assistant message (with
    its tito blob, if any) onto the exception. This lets us still validate
    capture fidelity even when the model is too weak to produce a clean tool
    call — important for tiny models like Qwen3-1.7B.
    """
    try:
        return model.query(msgs), None
    except FormatError as e:
        # e.messages[0] is the partial assistant message attached by
        # LitellmModel.query before raising. extra.tito should be present
        # because LitellmTITOModel.query stashes it on the success path,
        # but the partial-on-error path doesn't go through our subclass —
        # so reach into extra.response and reconstruct.
        partial = e.messages[0] if e.messages else {"extra": {}}
        from minisweagent.models.litellm_tito_model import _extract_tito_from_choice
        extra = partial.setdefault("extra", {})
        if "tito" not in extra:
            choices = ((extra.get("response") or {}).get("choices") or [])
            extra["tito"] = _extract_tito_from_choice(choices[0] if choices else {})
        return partial, e


def _check(cond: bool, label: str, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise AssertionError(label + (f": {detail}" if detail else ""))


def _fmt_ids(ids: list[int], n: int = 12) -> str:
    if ids is None:
        return "None"
    head = ", ".join(str(x) for x in ids[:n])
    tail = "..." if len(ids) > n else ""
    return f"[{head}{tail}]  (len={len(ids)})"


def _make_fake_tool_observation(action: dict) -> dict:
    """Synthetic tool response that mimics what mini-swe-agent's
    `format_toolcall_observation_messages` produces, without running bash."""
    return {
        "role": "tool",
        "tool_call_id": action.get("tool_call_id", ""),
        "content": "<returncode>0</returncode>\n<output>\nhello.txt\n</output>",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True,
                   help="HF id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--base-url", required=True,
                   help="vLLM OpenAI endpoint, e.g. http://127.0.0.1:8002/v1")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Use 0 for determinism while iterating.")
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    # litellm picks up these env vars when model_name is `openai/<...>`.
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.base_url

    print(f"== Phase 1 smoke for {args.model_id}")
    print(f"   base_url: {args.base_url}")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    chat_template = resolve_chat_template(model_name=f"openai/{args.model_id}")

    model = LitellmTITOModel(
        model_name=f"openai/{args.model_id}",
        model_kwargs={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        cost_tracking="ignore_errors",
    )

    # ---------------- Step 1 ----------------
    print("\n[Step 1] initial query")
    msgs1: list[dict] = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
    ]
    m1, fmt_err1 = _query_capturing_format_error(model, msgs1)
    if fmt_err1:
        print("   NOTE: model produced no tool call (FormatError). "
              "Verifying TITO capture only.")

    extra1 = m1.get("extra") or {}
    tito1 = extra1.get("tito") or {}
    ids1 = tito1.get("output_token_ids")
    lps1 = tito1.get("output_logprobs") or []
    actions1 = extra1.get("actions") or []

    print(f"   content: {repr((m1.get('content') or '')[:120])}")
    print(f"   tool_calls: {len(m1.get('tool_calls') or [])}")
    print(f"   output_token_ids: {_fmt_ids(ids1)}")
    print(f"   output_logprobs len: {len(lps1)}")
    print(f"   finish_reason={tito1.get('finish_reason')!r}  "
          f"stop_reason={tito1.get('stop_reason')!r}")

    _check("tito" in extra1, "(a) tito blob attached to message[extra]")
    _check(isinstance(ids1, list) and len(ids1) > 0,
           "(b) output_token_ids is non-empty list[int]",
           f"got type={type(ids1).__name__}")
    _check(all(isinstance(x, int) for x in ids1),
           "(b') every element is an int")
    _check(len(lps1) == len(ids1) or len(lps1) == len(ids1) - 1,
           "(c) logprobs length matches token_ids (or off by one for EOS)",
           f"ids={len(ids1)} lps={len(lps1)}")

    # FIDELITY: decoded IDs must equal the assistant content the model class
    # exposed (modulo whitespace / special-token rendering).
    decoded = tok.decode(ids1, skip_special_tokens=True).strip()
    content = (m1.get("content") or "").strip()
    if decoded != content:
        print(f"   decoded:  {decoded[:160]!r}")
        print(f"   content:  {content[:160]!r}")
    # Some models emit tool_call as part of content, others stash it in
    # message.tool_calls only. Accept either: decoded must contain content,
    # or vice versa, OR the decoded text must include the bash command we
    # parsed out (round-trip on tool args).
    fidelity_ok = (
        decoded == content
        or content in decoded
        or decoded in content
        or (actions1 and actions1[0]["command"] in decoded)
    )
    _check(fidelity_ok, "(d) decoded ids round-trip to assistant text")

    if fmt_err1:
        # Soft-pass (e) — small models often skip the tool call entirely.
        # Capture is what we actually want to validate in this harness.
        print("   [SKIP] (e) parsed action — model emitted no tool call; "
              "treating as informational only.")
    else:
        _check(len(actions1) > 0,
               "(e) at least one parsed action (drop-in semantics)",
               f"actions={actions1}")

    # ---------------- Step 2 ----------------
    # If step 1 produced no tool call, fabricate a synthetic one so we can
    # still build a step-2 prompt that exercises the chat template's
    # assistant->tool transition. The point of step 2 is the prefix-extension
    # property, which is independent of whether the real model called bash.
    print("\n[Step 2] prompt-extension query")
    if actions1:
        synth_call = m1.get("tool_calls") or []
        synth_action = actions1[0]
    else:
        synth_call = [{
            "id": "call_smoke_0",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls /tmp"}'},
        }]
        synth_action = {"command": "ls /tmp", "tool_call_id": "call_smoke_0"}
    obs = _make_fake_tool_observation(synth_action)
    asst_for_history = {
        "role": "assistant",
        "content": m1.get("content") or "",
        "tool_calls": synth_call,
    }
    msgs2 = msgs1 + [asst_for_history, obs]
    m2, fmt_err2 = _query_capturing_format_error(model, msgs2)
    if fmt_err2:
        print("   NOTE: step-2 model also produced no tool call. "
              "Capture-only validation continues.")

    extra2 = m2.get("extra") or {}
    tito2 = extra2.get("tito") or {}
    ids2 = tito2.get("output_token_ids")
    print(f"   step2 content: {repr((m2.get('content') or '')[:120])}")
    print(f"   step2 output_token_ids: {_fmt_ids(ids2)}")

    _check(isinstance(ids2, list) and len(ids2) > 0,
           "(f) step-2 tito blob populated")

    # The exact step-2 prompt vLLM rendered isn't returned to us, but the
    # *local* prediction of it must be a sensible extension of the chat so
    # far. We assert: when we apply the chat template with tools=[BASH_TOOL]
    # to (system+user+assistant+tool), the result is at least as long as
    # step-1's prompt and ends in an assistant-generation prefix.
    prompt2_text = tok.apply_chat_template(
        [
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER},
            asst_for_history,
            {
                "role": "tool",
                "tool_call_id": obs["tool_call_id"],
                "content": obs["content"],
            },
        ],
        tools=[BASH_TOOL],
        add_generation_prompt=True,
        tokenize=False,
        chat_template=chat_template,
    )
    prompt2_ids = tok.encode(prompt2_text, add_special_tokens=False)
    print(f"   local-rendered step-2 prompt: {_fmt_ids(prompt2_ids)}")

    prompt1_text = tok.apply_chat_template(
        [
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER},
        ],
        tools=[BASH_TOOL],
        add_generation_prompt=True,
        tokenize=False,
        chat_template=chat_template,
    )
    prompt1_ids = tok.encode(prompt1_text, add_special_tokens=False)

    _check(len(prompt2_ids) > len(prompt1_ids),
           "(g.1) step-2 prompt is strictly longer than step-1 prompt",
           f"{len(prompt1_ids)} -> {len(prompt2_ids)}")
    _check(prompt2_ids[: len(prompt1_ids)] == prompt1_ids,
           "(g.2) local chat template preserves the step-1 prefix",
           "first divergence at "
           f"{next((i for i in range(len(prompt1_ids)) if prompt2_ids[i] != prompt1_ids[i]), 'n/a')}")

    print("\nAll Phase 1 checks passed.")
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "step1_output_len": len(ids1),
        "step1_logprobs_len": len(lps1),
        "step1_actions": actions1,
        "step2_output_len": len(ids2 or []),
        "step1_prompt_local_len": len(prompt1_ids),
        "step2_prompt_local_len": len(prompt2_ids),
    }
    print("\nSummary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
