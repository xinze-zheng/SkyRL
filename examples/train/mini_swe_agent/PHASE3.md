# Phase 3 — TITO end-to-end smoke test

`phase3_smoke.py` is a lightweight reproducer that drives mini-swe-agent
through **one** SWE-Gym instance with TITO bookkeeping enabled and inspects
the resulting trajectory. Use it as a regression gate after touching:

- `minisweagent.models.litellm_tito_model.LitellmTITOModel`
- `minisweagent.agents.tito.{TITOAgent, TITOAgentState}`
- `examples/train/mini_swe_agent/mini_swe_generator.py`

Wall-clock: ~30 seconds on a small vLLM (Qwen3-1.7B) with the docker image
already cached.

---

## What it tests

| Check | What's being verified |
| --- | --- |
| `tito_payload returned from Ray task` | `init_and_run` propagates the agent's TITO state to the generator |
| `info.tito section present in saved JSON` | `TITOAgent.serialize()` round-trips correctly |
| `n_gen + n_obs == n_tokens` | Loss-mask partition is exact |
| `prompt_len + response_len == n_tokens` | Slicing at `prompt_len` is consistent |
| `tokens / loss_mask / logprobs lengths match n_tokens` | Three parallel arrays stay aligned through `absorb_step` |
| `sum(loss_mask) == n_gen` | All sampled tokens are labelled `1`; everything else `0` |
| `prompt region is fully masked-out` | Initial system+user prompt has no train-on labels |
| `step k prompt extends step k-1 prompt+output` | The strict-prefix invariant the agent uses to extract observation deltas |
| `payload prompt_len / tokens / logprobs match saved JSON` | Generator-bound payload is consistent with what was persisted |

A failure on **any** check tells you specifically which layer drifted.

---

## Prerequisites

### 1. SWE-Gym data (one-off)

```bash
cd SkyRL
uv run --isolated examples/train/mini_swe_agent/preprocess_swegym.py \
    --output_dir ~/wxzheng/data/swe_gym_subset
```

### 2. Docker image for the chosen instance

The agent runs the bash sandbox inside a SWE-Gym Docker container. Pick an
instance whose image you already have cached:

```bash
docker images | grep sweb.eval | head
# Example: xingyaoww/sweb.eval.x86_64.getmoto_s_moto-6190:latest  ← image present
```

The default `--instance-id getmoto__moto-6190` is one we've cached during
development. Any other cached instance also works.

### 3. vLLM server (Qwen3-1.7B is plenty for this test)

```bash
CUDA_VISIBLE_DEVICES=<idle GPU> nohup uv run vllm serve Qwen/Qwen3-1.7B \
    --port 8002 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.3 \
    > /tmp/vllm-phase3.log 2>&1 &

# Wait for "Application startup complete" in /tmp/vllm-phase3.log:
curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8002/v1/models
# Expect: HTTP 200
```

If `nvidia-smi` shows a previous SkyRL run still holding the GPU, free it
with `ray stop --force` (or kill the parent — see `ps -ef --forest`).

### 4. (Once) install pytest into the SkyRL venv if you also want unit tests

```bash
cd SkyRL && uv pip install pytest
```

---

## Run

```bash
cd SkyRL

uv run python examples/train/mini_swe_agent/phase3_smoke.py \
    --model-id Qwen/Qwen3-1.7B \
    --base-url http://127.0.0.1:8002/v1 \
    --instance-id getmoto__moto-6190 \
    --step-limit 4
```

Useful flags:

- `--keep-traj-dir /tmp/phase3_keep` — persist trajectory JSONs for
  inspection instead of writing to a tmpdir.
- `--max-tokens 256` — shorter per-call generation; fastest mode.
- `--step-limit 1` — collapses to "did one model call succeed and get
  recorded?". Skips the multi-turn prefix-invariant checks.

### Expected output (success)

```
== Phase 3 smoke for getmoto__moto-6190
   model: Qwen/Qwen3-1.7B    base_url: http://127.0.0.1:8002/v1
   trajectory dir: /tmp/phase3_xxxx
[run] launching init_and_run with use_tito=True ...
   messages: 8  reward: 0  error: '...'
   tito_payload present: True
  [PASS] info.tito section present in saved JSON
   n_steps=N  n_tokens=...  n_gen=...  n_obs=...  prompt_len=...  response_len=...
  [PASS] at least one TITO step recorded
  [PASS] n_gen + n_obs == n_tokens (mask partition)
  [PASS] prompt_len + response_len == n_tokens
  [PASS] tokens array length matches n_tokens
  [PASS] loss_mask array length matches n_tokens
  [PASS] logprobs array length matches n_tokens
  [PASS] sum(loss_mask) == n_gen
  [PASS] prompt region is fully masked-out (all 0s)
  [PASS] step 1 prompt extends step 0 prompt+output
  ...
   model class in saved config: minisweagent.models.litellm_tito_model.LitellmTITOModel
  [PASS] tito_payload returned from Ray task
  [PASS] payload prompt_len matches saved JSON
  [PASS] payload tokens length matches saved JSON
  [PASS] payload logprobs length matches saved JSON
All Phase 3 checks passed.
```

A non-zero exit code means at least one check failed; the failing labels
are listed at the bottom along with the path to the saved trajectory for
inspection.

---

## Known failure modes

### A. `TITO prefix invariant violated at step k` (Qwen3-only, by design)

The agent's `TITOAgentState.absorb_step` raises if the next-turn prompt
does NOT extend `(prev_prompt + prev_output)` byte-for-byte. With
**Qwen3 + `/v1/chat/completions`** this fires reliably around step 2-3
because Qwen3's chat template **strips `<think>...</think>` blocks from
non-final assistant turns** when re-rendering the conversation. The
template is position-dependent: thinking is preserved on the *current*
assistant turn but removed from earlier ones.

This is **not data corruption** — the tokens we recorded are exactly what
vLLM sampled, including the thinking. The mismatch is between the
*recorded* assistant span and the *re-rendered* assistant span, not
between the model's true output and our captured tokens.

For RL training, we **want** to keep the thinking tokens with `mask=1`
and real logprobs: that's the whole point of training a reasoning model.
The next-turn context-distribution shift (model thought during sampling
but sees no thinking in older turns at inference time) is a separate
training-vs-inference issue handled by the model itself, not by removing
tokens from the training stream.

If you hit this:

1. Confirm it's the Qwen3 thinking-strip by inspecting `transitions[k]`
   in the trajectory JSON: `prev_output` should start with token IDs
   `[151667, 198, 151668, ...]` (`<think>`, `\n`, `</think>`, ...) which
   are absent from `transitions[k+1].prompt_token_ids`.
2. The recorded `tokens / loss_mask / logprobs` arrays are still correct
   for training. The error is from the safety assert.
3. The proper fix (planned) is one of:
   - Switch `LitellmTITOModel` to `/v1/completions` and have the agent
     own the prompt as a token array end-to-end (no chat-template
     re-render between turns).
   - Relax the assert: trust the recorded tokens, accept context drift,
     and locate observation tokens by anchor-matching the generation
     prompt suffix at the end of `new_prompt` instead of by prefix
     equality.

### B. `Completions.create() got an unexpected keyword argument '...'`

`LitellmTITOModel` bypasses litellm and talks to the OpenAI SDK directly;
some YAML config keys are litellm-only and must be filtered. The model
class already strips a known list (`drop_params`, `num_retries`,
`api_base`, etc.). If a new one slips through, add it to
`LitellmTITOModel._LITELLM_ONLY_KWARGS`.

### C. `Free memory on device cuda:0 ... is less than desired`

Another process holds the GPU. Choose a different one
(`CUDA_VISIBLE_DEVICES=N`) or stop the existing user (`ray stop --force`,
then verify with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`).

### D. `error: unrecognized input` (in eval, not in TITO)

The generated patch was empty or malformed; `git apply` then `bash`
rejects it. This is a model-quality issue, not a TITO bug. Ignore for
the purposes of this smoke test — TITO checks still run.

### E. Trajectory written but `n_steps == 0`

The agent failed *before* the first model call could complete (e.g.,
model auth error, vLLM unreachable). Check:

```bash
tail -50 /tmp/vllm-phase3.log
curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8002/v1/models
```

---

## Why thinking tokens stay in the training stream

A natural reaction to the prefix-invariant failure is "let's normalize
the recorded output to match what vLLM re-renders next turn (i.e. drop
`<think>` from older turns in our recorded tokens too)." **Don't.**

- The model **sampled** those thinking tokens. Their logprobs are real.
  They came from the same forward pass as the tool call.
- Removing them removes the gradient signal that teaches the model to
  reason. Training only on the tool-call tokens would push the model
  toward "tool-call without reasoning", which is the opposite of what we
  want.
- The fact that vLLM hides the thinking from later prompts is a
  *prompt-rendering* concern, not a *what-to-train-on* concern. They are
  two different layers and should stay separate.

The strict prefix assert in `TITOAgentState` exists to protect against
*silent* drift between the recorded tokens and what the next prompt
contains — without it, observation deltas would be miscomputed. It's
intentionally loud so you notice when the layers diverge. The fix is to
change how observation deltas are extracted, not to hide the thinking
tokens.
