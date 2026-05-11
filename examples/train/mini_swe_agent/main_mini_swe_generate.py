"""
Generation-only entrypoint for mini-swe-agent. For dev/debugging purposes.

Auto-launches the SkyRL vLLM inference backend, runs a few rollouts through
MiniSweAgentGenerator.generate(), and prints results. No training loop.

Usage:
  # Prepare dataset first:
  uv run --isolated examples/train/mini_swe_agent/preprocess_swegym.py --output_dir ~/data/swe_gym_subset

  # Run generation only (uses the same CLI config as training):
  uv run --isolated --extra fsdp --extra miniswe --env-file examples/train/mini_swe_agent/.env.miniswe \
    -m examples.train.mini_swe_agent.main_mini_swe_generate \
    data.train_data="['~/data/swe_gym_subset/train.parquet']" \
    data.val_data="['~/data/swe_gym_subset/validation.parquet']" \
    trainer.policy.model.path="Qwen/Qwen3-8B" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.placement.policy_num_gpus_per_node=8 \
    trainer.placement.ref_num_gpus_per_node=8 \
    generator.inference_engine.num_engines=8 \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.enable_http_endpoint=true \
    generator.inference_engine.http_endpoint_host='127.0.0.1' \
    generator.inference_engine.http_endpoint_port=8001 \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.gpu_memory_utilization=0.8 \
    generator.batched=true \
    generator.n_samples_per_prompt=1 \
    generator.sampling_params.max_generate_length=2048 \
    generator.max_input_length=8192 \
    generator.max_turns=5 \
    trainer.train_batch_size=8 \
    trainer.logger=console \
    generator.miniswe_config_path="examples/train/mini_swe_agent/swebench.yaml" \
    generator.miniswe_traj_dir="/tmp/mini_swe_generate_trajs"

  # Or reuse the dev script config and just swap the entrypoint module.
"""

import asyncio
import os
import sys

import ray
from loguru import logger

from skyrl.train.config import SkyRLGymConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.generators.base import BatchMetadata, GeneratorInput, TrajectoryID
from skyrl.train.utils import initialize_ray

from .mini_swe_generator import MiniSWEGeneratorConfig, MiniSweAgentGenerator

MiniSWEConfig = make_config(generator_cls=MiniSWEGeneratorConfig)

NUM_SAMPLES_TO_TEST = 2


class MiniSWEGenerateExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return MiniSweAgentGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=SkyRLGymConfig(max_env_workers=0),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )

    def _setup_generator(self):
        logger.info("Setting up inference backend...")
        # For generation-only mode, we override get_inference_client behavior:
        # The base class sleeps engines after creation when colocate_all=true
        # (to free GPU for training). Since we have no trainer, we must skip that.
        # We call _get_new_inference_client() directly and skip the sleep.
        from skyrl.backends.skyrl_train.inference_servers.setup import build_new_inference_client

        is_colocated = self.cfg.trainer.placement.colocate_all
        client, server_setup = build_new_inference_client(
            self.cfg,
            self.tokenizer,
            placement_group=self.colocate_pg if is_colocated else None,
        )
        self._inference_router = server_setup.router
        self._server_groups = server_setup.server_groups
        self._prefill_server_groups = server_setup.prefill_server_groups
        self._decode_server_groups = server_setup.decode_server_groups
        # NOTE: intentionally skip client.sleep() — no trainer to reload weights from

        inference_engine_client = client

        # Point OPENAI_BASE_URL to the actual inference server URL so that
        # init_and_run Ray tasks (which use litellm/openai) can reach vLLM.
        if hasattr(inference_engine_client, "proxy_url"):
            actual_url = f"{inference_engine_client.proxy_url}/v1"
            os.environ["OPENAI_BASE_URL"] = actual_url
            logger.info(f"Set OPENAI_BASE_URL={actual_url}")

        logger.info("Inference backend ready.")
        return self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

    def run(self):
        generator = self._setup_generator()

        # Build input from the training dataset
        # PromptDataset.__getitem__ returns (messages, env_class, extras, uid)
        prompts = []
        env_extras = []
        trajectory_ids = []
        for i in range(min(NUM_SAMPLES_TO_TEST, len(self.train_dataset))):
            messages, env_class, extras, uid = self.train_dataset[i]
            prompts.append(messages)
            env_extras.append({"instance": extras["instance"], "data_source": extras["data_source"]})
            trajectory_ids.append(TrajectoryID(instance_id=extras["instance"]["instance_id"], repetition_id=0))

        n = len(prompts)
        logger.info(f"Running generation on {n} samples...")

        input_batch: GeneratorInput = {
            "prompts": prompts[:n],
            "env_classes": ["null"] * n,
            "env_extras": env_extras[:n],
            "sampling_params": None,
            "trajectory_ids": trajectory_ids[:n],
            "batch_metadata": BatchMetadata(global_step=0, training_phase="generate"),
        }

        output = asyncio.run(generator.generate(input_batch))

        # Print results
        logger.info("=" * 60)
        logger.info(f"Generated {len(output['response_ids'])} trajectories")
        logger.info(f"Rewards:      {output['rewards']}")
        logger.info(f"Stop reasons: {output['stop_reasons']}")
        logger.info(f"Prompt lens:  {[len(p) for p in output['prompt_token_ids']]}")
        logger.info(f"Response lens: {[len(r) for r in output['response_ids']]}")
        if output.get("rollout_metrics"):
            for k, v in output["rollout_metrics"].items():
                logger.info(f"  {k}: {v}")
        logger.info("=" * 60)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = MiniSWEGenerateExp(cfg)
    exp.run()


def main() -> None:
    cfg = MiniSWEConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
