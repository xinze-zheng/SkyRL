#!/usr/bin/env python
"""
Standalone single-rollout test exercising MiniSweAgentGenerator.generate().

Runs ONE SWE-bench instance through the full generator pipeline so you can
test changes to the inference server path end-to-end without launching the
full training loop.

Prerequisites:
  - vLLM server running on --port 8001 (or pass --port):
      python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-8B --port 8001 --gpu-memory-utilization 0.8
  - Docker accessible (sudo chmod 666 /var/run/docker.sock)
  - Dataset prepared: python examples/train/mini_swe_agent/preprocess_swegym.py

Usage:
  python examples/train/mini_swe_agent/test_single_rollout.py
  python examples/train/mini_swe_agent/test_single_rollout.py --index 5 --step-limit 3
  python examples/train/mini_swe_agent/test_single_rollout.py --model Qwen/Qwen3-1.7B --port 9000
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def build_generator_cfg(args):
    """Build a minimal GeneratorConfig for the test."""
    from skyrl.train.config import GeneratorConfig

    cfg = GeneratorConfig()
    cfg.inference_engine.enable_http_endpoint = True
    cfg.inference_engine.http_endpoint_host = args.host
    cfg.inference_engine.http_endpoint_port = args.port
    cfg.inference_engine.backend = "vllm"
    cfg.sampling_params.max_generate_length = args.max_generate_length
    cfg.max_input_length = args.max_input_length
    cfg.miniswe_config_path = args.config
    cfg.miniswe_traj_dir = args.output_dir
    return cfg


def build_generator_input(row, args):
    """Build a GeneratorInput dict for a single instance."""
    from skyrl.train.generators.base import BatchMetadata, GeneratorInput, TrajectoryID

    instance = row["instance"]
    data_source = row["data_source"]

    generator_input: GeneratorInput = {
        "prompts": [row["prompt"]],
        "env_classes": ["null"],
        "env_extras": [{"instance": instance, "data_source": data_source}],
        "sampling_params": None,
        "trajectory_ids": [TrajectoryID(instance_id=instance["instance_id"], repetition_id=0)],
        "batch_metadata": BatchMetadata(global_step=0, training_phase="eval"),
    }
    return generator_input


def main():
    parser = argparse.ArgumentParser(description="Single rollout via MiniSweAgentGenerator")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/data/swe_gym_subset"))
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--index", type=int, default=0, help="Dataset row index")
    parser.add_argument("--config", default="examples/train/mini_swe_agent/swebench.yaml")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--output-dir", default="/tmp/mini_swe_test_traj")
    parser.add_argument("--step-limit", type=int, default=None, help="Override agent step limit")
    parser.add_argument("--max-generate-length", type=int, default=2048)
    parser.add_argument("--max-input-length", type=int, default=8192)
    args = parser.parse_args()

    # Set env vars for litellm/openai
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")
    os.environ["OPENAI_BASE_URL"] = f"http://{args.host}:{args.port}/v1"
    os.environ["LITELLM_MODEL_REGISTRY_PATH"] = "examples/train/mini_swe_agent/litellm.json"
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

    # Optionally patch step_limit in the swebench config
    if args.step_limit is not None:
        import yaml
        from minisweagent.config import get_config_path

        config_path = get_config_path(args.config)
        sweagent_config = yaml.safe_load(config_path.read_text())
        sweagent_config.setdefault("agent", {})["step_limit"] = args.step_limit
        # Write to a temp file and override
        tmp_config = Path(args.output_dir) / "swebench_override.yaml"
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        tmp_config.write_text(yaml.dump(sweagent_config))
        args.config = str(tmp_config)
        logger.info(f"Overriding step_limit to {args.step_limit}")

    import ray
    from skyrl.train.config import SkyRLGymConfig
    from skyrl.utils.tok import get_tokenizer

    from examples.train.mini_swe_agent.mini_swe_generator import (
        MiniSweAgentGenerator,
    )

    # Init Ray (needed for init_and_run remote task)
    ray.init(ignore_reinit_error=True)

    # Load dataset
    parquet_path = os.path.join(args.data_dir, f"{args.split}.parquet")
    logger.info(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    row = df.iloc[args.index].to_dict()
    instance_id = row["instance"]["instance_id"]
    logger.info(f"Instance: {instance_id} (index={args.index})")

    # Build generator config
    generator_cfg = build_generator_cfg(args)

    # Load tokenizer
    tokenizer = get_tokenizer(args.model)

    # Create generator with a mock inference_engine_client
    # (the generator never calls it — mini-swe-agent talks to vLLM directly via HTTP)
    mock_client = MagicMock()
    generator = MiniSweAgentGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=SkyRLGymConfig(max_env_workers=0),
        inference_engine_client=mock_client,
        tokenizer=tokenizer,
        model_name=args.model,
    )
    logger.info(f"Generator created, model={args.model}, endpoint={os.environ['OPENAI_BASE_URL']}")

    # Build input and run generate()
    generator_input = build_generator_input(row, args)
    logger.info("Running generator.generate() ...")

    output = asyncio.run(generator.generate(generator_input))

    # Print results
    print("\n" + "=" * 60)
    print(f"Instance:       {instance_id}")
    print(f"Reward:         {output['rewards']}")
    print(f"Stop reasons:   {output['stop_reasons']}")
    print(f"Prompt len:     {[len(p) for p in output['prompt_token_ids']]}")
    print(f"Response len:   {[len(r) for r in output['response_ids']]}")
    if output.get("rollout_metrics"):
        for k, v in output["rollout_metrics"].items():
            print(f"  {k}: {v}")
    print(f"Trajectory dir: {args.output_dir}")
    print("=" * 60)

    ray.shutdown()


if __name__ == "__main__":
    main()
