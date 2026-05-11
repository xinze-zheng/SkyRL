set -x

# Adapted GRPO training for Qwen3-1.7B on SWE-Bench.
# Uses mini-swe-agent v2 with native tool calling.
# Designed for 4x 22GB GPUs (e.g., A10G).
#
# 1. Prepare dataset:
#    uv run --isolated examples/train/mini_swe_agent/preprocess_swegym.py --output_dir ~/data/swe_gym_subset
#
# 2. Ensure Docker is running:
#    docker info
#
# 3. Launch training:
#    bash examples/train/mini_swe_agent/run_mini_swe_qwen3_1.7B.sh

DATA_DIR="$HOME/data/swe_gym_subset"
MINISWE_TRAJ_DIR="$HOME/mini_swe_agent_trajs"

: "${NUM_GPUS:=4}"
: "${LOGGER:=console}"

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export LITELLM_MODEL_REGISTRY_PATH=examples/train/mini_swe_agent/litellm.json
export MSWEA_COST_TRACKING=ignore_errors

export RAY_memory_usage_threshold=0.99

set -a && source examples/train/mini_swe_agent/.env.miniswe && set +a
python -m examples.train.mini_swe_agent.main_mini_swe \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-1.7B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=5 \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=2048 \
  generator.max_input_length=8192 \
  generator.max_turns=10 \
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
  generator.n_samples_per_prompt=2 \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes \
  generator.inference_engine.engine_init_kwargs.chat_template="$(pwd)/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2" \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe" \
  trainer.run_name="mini_swe_qwen3_1.7B" \
  trainer.resume_mode=null \
  generator.miniswe_config_path="examples/train/mini_swe_agent/swebench.yaml" \
  generator.miniswe_traj_dir="$MINISWE_TRAJ_DIR" \
  $@
