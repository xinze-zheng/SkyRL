"""Phase 3 smoke test for TITO bookkeeping.

Drives mini-swe-agent through ONE SWE-Gym instance with TITO enabled, then
inspects the resulting `info.tito.*` block in the saved trajectory JSON.

This is a deliberately minimal end-to-end check:

  * No Ray cluster — the @ray.remote function is unwrapped and called in
    the current process so tracebacks are clean.
  * No SkyRL trainer — only the agent loop and saved trajectory are
    exercised.
  * Default `--step-limit 4` keeps wall-clock under ~30s on cached
    SWE-Gym docker images and a small vLLM (1.7B).

Use it as a regression gate after any change to:

  * `minisweagent.models.litellm_tito_model.LitellmTITOModel`
  * `minisweagent.agents.tito.{TITOAgent, TITOAgentState}`
  * `examples.train.mini_swe_agent.mini_swe_generator.{TITOAgentWithReminder,
    init_and_run}`

See `examples/train/mini_swe_agent/PHASE3.md` for prerequisites,
expected output, and known-failure modes (Qwen3 `<think>`-strip, etc.).

Usage:
  cd SkyRL
  uv run python examples/train/mini_swe_agent/phase3_smoke.py \
      --instance-id getmoto__moto-6190
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Allow `python examples/train/mini_swe_agent/phase3_smoke.py` from the
# repo root: add the repo root to sys.path so `examples.*` is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import yaml

from skyrl.train.generators.base import TrajectoryID

from examples.train.mini_swe_agent.mini_swe_generator import (
    MiniSWEGeneratorConfig,
    init_and_run,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_instance(parquet_path: str, instance_id: str | None) -> dict:
    df = pd.read_parquet(parquet_path)
    if instance_id is None:
        row = df.iloc[0]
    else:
        hits = df[df["instance_id"] == instance_id]
        if hits.empty:
            raise SystemExit(f"instance_id {instance_id} not in {parquet_path}")
        row = hits.iloc[0]
    inst = dict(row["instance"])
    for k, v in list(inst.items()):
        if hasattr(v, "tolist"):  # numpy → plain python
            inst[k] = v.tolist()
    return {
        "instance_id": row["instance_id"],
        "problem_statement": row["problem_statement"],
        "data_source": row["data_source"],
        "instance": inst,
    }


def _build_generator_cfg(traj_dir: Path) -> MiniSWEGeneratorConfig:
    cfg = MiniSWEGeneratorConfig()
    cfg.miniswe_traj_dir = str(traj_dir)
    cfg.miniswe_use_tito = True
    return cfg


class _Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, cond: bool, label: str, detail: str = "") -> None:
        mark = "PASS" if cond else "FAIL"
        line = f"  [{mark}] {label}" + (f" — {detail}" if detail else "")
        print(line)
        if not cond:
            self.failures.append(label + (f": {detail}" if detail else ""))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet",
        default="/home/uccl/wxzheng/data/swe_gym_subset/train.parquet",
        help="Path to a SWE-Gym parquet produced by preprocess_swegym.py.",
    )
    p.add_argument(
        "--instance-id",
        default="getmoto__moto-6190",
        help="Instance whose docker image is already cached locally. "
             "Pick one you have via `docker images | grep sweb.eval`.",
    )
    p.add_argument(
        "--config-path",
        default="examples/train/mini_swe_agent/swebench.yaml",
    )
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    p.add_argument(
        "--step-limit",
        type=int,
        default=4,
        help="Override agent.step_limit so the smoke run finishes fast.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Per-call generation cap. Lower = faster.",
    )
    p.add_argument(
        "--keep-traj-dir",
        default=None,
        help="Persist trajectory JSONs here (else a tmpdir is used).",
    )
    args = p.parse_args()

    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.base_url

    payload = _load_instance(args.parquet, args.instance_id)
    print(f"== Phase 3 smoke for {payload['instance_id']}")
    print(f"   model: {args.model_id}    base_url: {args.base_url}")

    sweagent_config = yaml.safe_load(Path(args.config_path).read_text())
    sweagent_config.setdefault("agent", {})["step_limit"] = args.step_limit
    sweagent_config.setdefault("agent", {})["cost_limit"] = 0.0
    sweagent_config.setdefault("model", {})
    sweagent_config["model"]["model_kwargs"] = (
        sweagent_config["model"].get("model_kwargs") or {}
    )
    sweagent_config["model"]["model_kwargs"].update({
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    })

    if args.keep_traj_dir:
        traj_dir = Path(args.keep_traj_dir)
        traj_dir.mkdir(parents=True, exist_ok=True)
    else:
        traj_dir = Path(tempfile.mkdtemp(prefix="phase3_"))
    print(f"   trajectory dir: {traj_dir}")

    cfg = _build_generator_cfg(traj_dir)
    bare_init_and_run = init_and_run._function   # bypass Ray

    print("\n[run] launching init_and_run with use_tito=True ...")
    messages, reward, error, tito_payload = bare_init_and_run(
        payload["instance"] | {
            "instance_id": payload["instance_id"],
            "problem_statement": payload["problem_statement"],
        },
        f"openai/{args.model_id}",
        sweagent_config,
        cfg,
        payload["data_source"],
        {},
        TrajectoryID(0, 0),
        global_step=0,
        training_phase="train",
        use_tito=True,
    )

    print(f"   messages: {len(messages)}  reward: {reward}  error: {error!r}")
    print(f"   tito_payload present: {tito_payload is not None}")

    saved_files = sorted((traj_dir / "step_0" / "train").glob("*.json"))
    if not saved_files:
        print("FAIL: no trajectory JSON written", file=sys.stderr)
        return 1
    saved = json.loads(saved_files[0].read_text())
    info = saved.get("info", {})

    chk = _Checker()
    chk.check("tito" in info, "info.tito section present in saved JSON")
    if "tito" not in info:
        return 1
    tito = info["tito"]
    n_tokens = tito["n_tokens"]
    n_gen = tito["n_gen_tokens"]
    n_obs = tito["n_obs_tokens"]
    prompt_len = tito["prompt_len"]
    response_len = tito["response_len"]
    n_steps = tito["n_steps"]

    print(
        f"\n   n_steps={n_steps}  n_tokens={n_tokens}  n_gen={n_gen}  "
        f"n_obs={n_obs}  prompt_len={prompt_len}  response_len={response_len}"
    )

    chk.check(n_steps >= 1, "at least one TITO step recorded",
              f"n_steps={n_steps}")
    chk.check(n_gen + n_obs == n_tokens,
              "n_gen + n_obs == n_tokens (mask partition)",
              f"{n_gen} + {n_obs} != {n_tokens}")
    chk.check(prompt_len + response_len == n_tokens,
              "prompt_len + response_len == n_tokens",
              f"{prompt_len} + {response_len} != {n_tokens}")
    chk.check(len(tito["tokens"]) == n_tokens,
              "tokens array length matches n_tokens")
    chk.check(len(tito["loss_mask"]) == n_tokens,
              "loss_mask array length matches n_tokens")
    chk.check(len(tito["logprobs"]) == n_tokens,
              "logprobs array length matches n_tokens")
    chk.check(sum(tito["loss_mask"]) == n_gen,
              "sum(loss_mask) == n_gen",
              f"{sum(tito['loss_mask'])} != {n_gen}")
    chk.check(all(m == 0 for m in tito["loss_mask"][:prompt_len]),
              "prompt region is fully masked-out (all 0s)")

    transitions = tito["transitions"]
    for i in range(1, len(transitions)):
        prev = transitions[i - 1]
        cur = transitions[i]
        expected_prefix = prev["prompt_token_ids"] + prev["output_token_ids"]
        chk.check(
            cur["prompt_token_ids"][: len(expected_prefix)] == expected_prefix,
            f"step {i} prompt extends step {i-1} prompt+output",
        )

    print(f"\n   model class in saved config: "
          f"{info.get('config', {}).get('model_type')}")

    chk.check(tito_payload is not None,
              "tito_payload returned from Ray task")
    if tito_payload is not None:
        chk.check(tito_payload["prompt_len"] == prompt_len,
                  "payload prompt_len matches saved JSON")
        chk.check(len(tito_payload["tokens"]) == n_tokens,
                  "payload tokens length matches saved JSON")
        chk.check(len(tito_payload["logprobs"]) == n_tokens,
                  "payload logprobs length matches saved JSON")

    if chk.failures:
        print(f"\n{len(chk.failures)} check(s) FAILED:")
        for f in chk.failures:
            print(f"  - {f}")
        print(f"\nSaved trajectory for inspection: {saved_files[0]}")
        return 1

    print("\nAll Phase 3 checks passed.")
    print(f"Saved trajectory: {saved_files[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
