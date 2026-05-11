set -x

# Generation-only dev script for mini-swe-agent.
# Auto-launches vLLM backend, runs a few rollouts, prints results. No training.
#
# Usage:
#   bash examples/train/mini_swe_agent/run_mini_swe_generate.sh
#   # Use a smaller/faster model for dev:
#   MODEL=Qwen/Qwen3-1.7B NUM_GPUS=4 bash examples/train/mini_swe_agent/run_mini_swe_generate.sh
#   # Override any param:
#   bash examples/train/mini_swe_agent/run_mini_swe_generate.sh generator.max_turns=3

DATA_DIR="$HOME/data/swe_gym_subset"
MINISWE_TRAJ_DIR="$HOME/mini_swe_agent_trajs"

: "${MODEL:=Qwen/Qwen3-8B}"
: "${NUM_GPUS:=8}"
LOGGER=console

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export LITELLM_MODEL_REGISTRY_PATH=examples/train/mini_swe_agent/litellm.json
export MSWEA_COST_TRACKING=ignore_errors

set -a && source examples/train/mini_swe_agent/.env.miniswe && set +a
python -m examples.train.mini_swe_agent.main_mini_swe_generate \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=2048 \
  generator.max_input_length=8192 \
  generator.max_turns=5 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host='127.0.0.1' \
  generator.inference_engine.http_endpoint_port=8001 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe_generate" \
  trainer.run_name="mini_swe_generate_dev" \
  trainer.resume_mode=null \
  generator.miniswe_config_path="examples/train/mini_swe_agent/swebench.yaml" \
  generator.miniswe_traj_dir="$MINISWE_TRAJ_DIR" \
  $@
