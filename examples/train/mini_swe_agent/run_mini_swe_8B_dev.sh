set -x

# Minimal dev config for fast iteration on Qwen3-8B + SWE-Bench.
# Differences from run_mini_swe_8B.sh:
#   - 1 epoch, tiny batches (2 train, 2 eval)
#   - No eval before train, no checkpointing
#   - Fewer inference engines (2) for faster startup
#   - Smaller sequence lengths
#   - 1 sample per prompt
#
# Usage:
#   bash examples/train/mini_swe_agent/run_mini_swe_8B_dev.sh
#   # Override anything via CLI args:
#   bash examples/train/mini_swe_agent/run_mini_swe_8B_dev.sh trainer.epochs=3

DATA_DIR="$HOME/data/swe_gym_subset"
MINISWE_TRAJ_DIR="$HOME/mini_swe_agent_trajs"

NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=8
TP_SIZE=1
LOGGER=console

uv run --isolated --extra fsdp --extra miniswe --env-file examples/train/mini_swe_agent/.env.miniswe -m examples.train.mini_swe_agent.main_mini_swe \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=1 \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.epochs=1 \
  trainer.eval_batch_size=2 \
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
  generator.inference_engine.engine_init_kwargs.chat_template="$(pwd)/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2" \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe_dev" \
  trainer.run_name="mini_swe_8B_dev" \
  trainer.resume_mode=null \
  generator.miniswe_config_path="examples/train/mini_swe_agent/swebench.yaml" \
  generator.miniswe_traj_dir="$MINISWE_TRAJ_DIR" \
  $@
