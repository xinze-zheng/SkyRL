import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import yaml
import traceback
import ray
from pathlib import Path

from minisweagent.models import get_model
from minisweagent.agents.default import DefaultAgent
from minisweagent.agents.tito import TITOAgent
from minisweagent.config import get_config_path
from .mini_swe_utils import evaluate_trajectory, get_sb_environment

from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl.train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)


@dataclass
class MiniSWEGeneratorConfig(GeneratorConfig):
    """Extended generator config with Mini-SWE-Agent-specific fields."""

    miniswe_config_path: str = ""
    miniswe_traj_dir: str = ""
    # When true, use the TITO agent + model: per-step token IDs and
    # logprobs are captured at request time; the generator reads them
    # straight off the agent state without re-tokenizing the messages.
    # Requires `model.model_type` in swebench.yaml to be set to
    # `minisweagent.models.litellm_tito_model.LitellmTITOModel`.
    miniswe_use_tito: bool = False


class _ReminderMixin:
    """Append per-step reminders to observation messages — same behavior as
    the original `DefaultAgentWithReminder`. Factored out so it can be mixed
    into both the default and TITO agents."""

    def execute_actions(self, message: dict) -> list:
        result = super().execute_actions(message)
        remaining = self.config.step_limit - self.n_calls
        if remaining == 1:
            reminder = "\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            reminder = f"\nREMINDER: You have {remaining} turns left to arrive at the solution."
        else:
            reminder = ""
        if result and reminder:
            result[-1]["content"] = result[-1].get("content", "") + reminder
        return result


class DefaultAgentWithReminder(_ReminderMixin, DefaultAgent):
    pass


class TITOAgentWithReminder(_ReminderMixin, TITOAgent):
    pass


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: GeneratorConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
    use_tito: bool = False,
):
    from loguru import logger

    model_config = sweagent_config.get("model", {})
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    model_config.setdefault("model_kwargs", {}).update(sampling_params)
    if use_tito:
        # Force the TITO-capable model class regardless of what's in the
        # YAML so the toggle is a single boolean from the run script.
        model_config["model_class"] = (
            "minisweagent.models.litellm_tito_model.LitellmTITOModel"
        )
    model = get_model(litellm_model_name, model_config)

    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    exit_status = None
    submission = None
    agent_cls = TITOAgentWithReminder if use_tito else DefaultAgentWithReminder
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = agent_cls(model, env, **sweagent_config.get("agent", {}))
        run_result = agent.run(instance["problem_statement"])
        exit_status = run_result.get("exit_status", "")
        submission = run_result.get("submission", "")
        result = submission
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status = type(e).__name__
        result = str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        path = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        path = path / filename
        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, sweagent_config, data_source)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}")
                logger.debug(f"traceback: {traceback.format_exc()}")
                eval_error = str(e)
                error = str(e)

            agent.save(path, {"info": {"submission": result, "reward": reward, "eval_error": eval_error}})

    # Return tito_state along with messages so the generator can consume
    # train-ready arrays directly. tito_state is None for non-TITO agents.
    tito_state = getattr(agent, "tito_state", None) if agent is not None else None
    tito_payload = None
    if tito_state is not None and tito_state.tokens:
        tito_payload = {
            "tokens": tito_state.tokens,
            "loss_mask": tito_state.loss_mask,
            "logprobs": tito_state.logprobs,
            "prompt_len": len(tito_state.prompt_ids()),
        }
    return (agent.messages if agent is not None else [], reward, error, tito_payload)


class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):

        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)

        self.http_server_inference_engine_client_host = generator_cfg.inference_engine.http_endpoint_host

        self.http_server_inference_engine_client_port = generator_cfg.inference_engine.http_endpoint_port

        self.base_url = (
            f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSWEAgentGenerator doesn't support custom chat template")

    def _build_output_from_tito(
        self,
        tito_payload: Dict[str, Any],
        *,
        max_tokens: int,
        max_input_length: int,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[float]]]:
        """Convert the agent-side TITO state into the generator return shape.

        TITO already gives us the exact tokens vLLM saw and sampled, the
        loss mask, and per-sampled-token logprobs — no re-tokenization
        needed.

        Truncation honors the same budget as the re-tokenization path:
            max_response_tokens = max_tokens + max_input_length - prompt_len
        Slicing tokens / loss_mask / logprobs uses the same index so the
        three arrays stay length-aligned.
        """
        prompt_len = int(tito_payload["prompt_len"])
        all_tokens = list(tito_payload["tokens"])
        all_loss_mask = list(tito_payload["loss_mask"])
        all_logprobs = list(tito_payload["logprobs"])

        prompt_ids = all_tokens[:prompt_len]
        response_ids = all_tokens[prompt_len:]
        loss_mask = all_loss_mask[prompt_len:]
        logprobs = all_logprobs[prompt_len:]

        max_response_tokens = max_tokens + max_input_length - prompt_len
        stop_reason = "complete"
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        logprobs = logprobs[:max_response_tokens]

        # Reward will be filled in by the caller (we don't have it here).
        # Use 0.0 as a placeholder; minisweagent_agent_loop overrides via
        # the wrapping return tuple in the existing code path. To preserve
        # the contract we must return the reward too — pull from outer
        # scope by returning None here and letting the caller substitute.
        # The caller (`minisweagent_agent_loop`) substitutes `reward` into
        # tuple position 1. Here we return a placeholder 0.0 that the
        # caller doesn't use because it builds its own tuple.
        return (response_ids, 0.0, stop_reason, loss_mask, prompt_ids, logprobs)

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:

        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        use_tito = bool(getattr(self.generator_cfg, "miniswe_use_tito", False))
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        messages, reward, error, tito_payload = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
            use_tito,
        )
        if not len(messages):
            return None, None, None, None, None, None

        # ------------------------------------------------------------------
        # TITO fast path: agent already produced train-ready tokens.
        # ------------------------------------------------------------------
        if use_tito and tito_payload is not None:
            response_ids, _placeholder_reward, stop_reason, loss_mask, prompt_ids, logprobs = (
                self._build_output_from_tito(
                    tito_payload,
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                )
            )
            return (response_ids, reward, stop_reason, loss_mask, prompt_ids, logprobs)

        # ------------------------------------------------------------------
        # Re-tokenization fallback (original path).
        # ------------------------------------------------------------------
        # Separate prompt messages (system + user) from response messages.
        # v2 messages may include roles: system, user, assistant, tool, exit
        prompt_end = 0
        for i, msg in enumerate(messages):
            if msg["role"] in ("system", "user") and i == prompt_end:
                prompt_end = i + 1
            else:
                break

        response_messages = messages[prompt_end:]

        initial_input_ids = self.tokenizer.apply_chat_template(
            messages[:prompt_end], add_generation_prompt=False, return_dict=False, tokenize=True
        )
        initial_prompt_length = len(initial_input_ids)

        # Remove trailing non-assistant messages (exit messages, trailing user/tool messages)
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] != "assistant":
            last_idx -= 1
        if last_idx < 0:
            # No assistant messages found — agent failed before generating any response
            return None, None, None, None, None, None
        response_messages = response_messages[: last_idx + 1]

        # For tokenization, convert tool messages to user messages since
        # get_response_ids_and_loss_mask_from_messages only handles user/assistant roles.
        # Tool messages should be masked (loss=0) just like user messages.
        normalized_messages = []
        for msg in response_messages:
            if msg["role"] == "tool":
                normalized_messages.append({"role": "user", "content": msg.get("content", "")})
            elif msg["role"] in ("assistant", "user"):
                normalized_messages.append({"role": msg["role"], "content": msg.get("content", "")})
            # Skip exit or other roles

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            normalized_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )

        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.inference_engine.backend, self.generator_cfg.sampling_params
        )

        tasks = []

        for i in range(len(prompts)):
            tasks.append(
                self.minisweagent_agent_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            )

        all_outputs = await asyncio.gather(*tasks)

        # Replace None entries (failed trajectories) with dummy outputs that have loss_mask=[0]
        # so they don't contribute to training but keep the count matching num_prompts.
        responses = []
        rewards = []
        stop_reasons = []
        loss_masks = []
        prompt_token_ids = []
        rollout_logprobs: list[Optional[list]] = []
        for output in all_outputs:
            if output[0] is not None:
                responses.append(output[0])
                rewards.append(output[1])
                stop_reasons.append(output[2])
                loss_masks.append(output[3])
                prompt_token_ids.append(output[4])
                rollout_logprobs.append(output[5])
            else:
                # Dummy entry: single token, zero reward, masked out
                responses.append([0])
                rewards.append(0.0)
                stop_reasons.append("error")
                loss_masks.append([0])
                prompt_token_ids.append([0])
                rollout_logprobs.append(None)

        if all(lm == [0] for lm in loss_masks):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        # Only emit logprobs if every trajectory has them (TITO mode).
        # Mixing real logprobs with None/synthetic ones would silently bias
        # the policy update; better to drop them than to corrupt training.
        emit_logprobs = all(lp is not None for lp in rollout_logprobs)
        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs if emit_logprobs else None,
        }

        return generator_output
