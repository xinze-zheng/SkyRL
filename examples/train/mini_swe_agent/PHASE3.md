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
| `transition observation_token_ids recorded` | Later-turn observations are tracked by fixed-base local tokenization |
| `tokens/loss_mask/logprobs reconstruct from transitions` | Saved arrays match `initial prompt + observation_token_ids + output_token_ids` |
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
    --trust-request-chat-template \
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
  recorded?". Multi-turn reconstruction checks still run when more than one
  step is recorded.

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
  [PASS] transition observation_token_ids recorded
  [PASS] tokens reconstruct from prompt + observations + outputs
  [PASS] loss_mask reconstructs from transition boundaries
  [PASS] logprobs reconstruct from output logprobs and observation padding
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

### A. `Chat template is passed with request, but --trust-request-chat-template is not set`

`LitellmTITOModel` sends the SkyRL-compatible Qwen3 chat template in the
request so vLLM does not strip prior thinking content while rendering chat
history. Restart vLLM with:

```bash
--trust-request-chat-template
```

### B. Prefix-invariant failures in older scripts

The old prefix-invariant test is obsolete for fixed-base TITO. The current
Phase 3 smoke reconstructs the saved arrays from
`prompt_token_ids + observation_token_ids + output_token_ids` instead. If an
older helper still checks that each vLLM-rendered prompt extends the previous
prompt and output byte-for-byte, treat it as diagnostic only, not a correctness
gate.

### C. `Completions.create() got an unexpected keyword argument '...'`

`LitellmTITOModel` bypasses litellm and talks to the OpenAI SDK directly;
some YAML config keys are litellm-only and must be filtered. The model
class already strips a known list (`drop_params`, `num_retries`,
`api_base`, etc.). If a new one slips through, add it to
`LitellmTITOModel._LITELLM_ONLY_KWARGS`.

### D. `Free memory on device cuda:0 ... is less than desired`

Another process holds the GPU. Choose a different one
(`CUDA_VISIBLE_DEVICES=N`) or stop the existing user (`ray stop --force`,
then verify with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`).

### E. `error: unrecognized input` (in eval, not in TITO)

The generated patch was empty or malformed; `git apply` then `bash`
rejects it. This is a model-quality issue, not a TITO bug. Ignore for
the purposes of this smoke test — TITO checks still run.

### F. Trajectory written but `n_steps == 0`

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

Fixed-base TITO keeps these layers separate: sampled assistant tokens come
from vLLM, while later observation tokens are computed locally from a small
fixed-base conversation. This is why the Phase 3 smoke now validates array
reconstruction from transitions instead of cross-turn vLLM prompt prefix
equality.
