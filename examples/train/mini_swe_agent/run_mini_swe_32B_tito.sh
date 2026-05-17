set -x

# TITO + Renderer training for Qwen3-32B on SWE-Bench.
# Single node, 8 GPUs (B200, 183GB each), colocated training+generation.
# Qwen3-32B is dense (~64GB bf16), fits in 1 B200 GPU for inference (TP=1).
# FSDP shards training across all 8 GPUs.
#
# Prerequisites:
#   python examples/train/mini_swe_agent/preprocess_swegym.py --output_dir ~/data/swe_gym_subset
#
# Usage:
#   PATH=$PWD/.venv/bin:$PATH bash examples/train/mini_swe_agent/run_mini_swe_32B_tito.sh
#   # Override any param:
#   bash examples/train/mini_swe_agent/run_mini_swe_32B_tito.sh trainer.epochs=5

DATA_DIR="$HOME/data/swe_gym_subset"
CKPT_PATH="$HOME/ckpts/llm_mini_swe_32B_tito"
MINISWE_TRAJ_DIR="$HOME/mini_swe_agent_trajs"

NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=8
TP_SIZE=1
LOGGER=console

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export LITELLM_MODEL_REGISTRY_PATH=examples/train/mini_swe_agent/litellm.json
export MSWEA_COST_TRACKING=ignore_errors

set -a && source examples/train/mini_swe_agent/.env.miniswe && set +a
python -m examples.train.mini_swe_agent.main_mini_swe \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-32B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=1 \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.epochs=3 \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=false \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=40960 \
  generator.max_turns=15 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.max_seq_len=40960 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host='127.0.0.1' \
  generator.inference_engine.http_endpoint_port=8001 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=1 \
  generator.inference_engine.gpu_memory_utilization=0.9 \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes \
  generator.inference_engine.engine_init_kwargs.chat_template="$(pwd)/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2" \
  generator.inference_engine.engine_init_kwargs.max_model_len=40960 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe_tito" \
  trainer.run_name="mini_swe_qwen3_32B_tito" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.miniswe_config_path="examples/train/mini_swe_agent/swebench.yaml" \
  generator.miniswe_traj_dir="$MINISWE_TRAJ_DIR" \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.tito.enabled=true \
  generator.inference_engine.tito.use_renderer=true \
  generator.inference_engine.tito.renderer_name=qwen3 \
  generator.inference_engine.tito.log_path="$MINISWE_TRAJ_DIR/tito_logs" \
  $@
